"""
Calibrate team strengths so the simulated group stage matches an external
prognosis (group_stage_prognosis.csv).

Model (from simulate_match.py): for a match between A and B the Poisson goal
rates are
    lambda_a = NORMAL_GOALS * strength_a / (strength_a + strength_b)
    lambda_b = NORMAL_GOALS * strength_b / (strength_a + strength_b)
Only relative strength matters and there is no home advantage. NORMAL_GOALS is
kept fixed; only the team strengths are calibrated.

Calibration is a simple gradient-free fixed-point loop: each iteration we
simulate every group, measure each team's expected finishing position, and nudge
its strength so a team finishing lower than its target gets stronger:
    strength *= exp(LR * (E_sim - E_target))
This converges because a team's expected finish is monotonic in its own
strength.

NOTE: The groups are independent in this calibration. Hence, it cannot say anything on the strength of teams in different groups.
"""

import csv
import json
from pathlib import Path

import numpy as np

UNIFORM_START_STRENGTH = 100.0

HERE = Path(__file__).parent
PARAMS_IN = HERE / "parameters_baseline.json"
PARAMS_OUT = HERE / "parameters.json"
GROUPS_FILE = HERE / "group_stage.json"
PROGNOSIS_FILE = HERE / "group_stage_prognosis.csv"

# Calibration hyper-parameters.
N_ITERS = 60
N_SIMS = 10_000  # simulations per group per iteration
N_SIMS_FINAL = 50_000  # simulations for the final report
LR = 0.05  # learning rate for the strength update
SEED = 12345


def load_targets() -> dict[str, np.ndarray]:
    """Parse the prognosis CSV into {team: [p1, p2, p3, p4]} (probabilities)."""
    targets: dict[str, np.ndarray] = {}
    with PROGNOSIS_FILE.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            team = row["Land"]
            probs = np.array(
                [
                    float(row[c].rstrip("%")) / 100.0
                    for c in ("1. plass", "2. plass", "3. plass", "4. plass")
                ]
            )
            targets[team] = probs / probs.sum()  # guard against rounding
    return targets


def simulate_group(
    matches: list[dict],
    teams: list[str],
    strengths: dict[str, float],
    normal_goals: float,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate a single group n_sims times.

    Returns a (4, 4) array `placement[t, p]` = fraction of sims where team
    `teams[t]` finished in placement `p` (0 = 1st ... 3 = 4th).
    """
    idx = {team: i for i, team in enumerate(teams)}
    points = np.zeros((4, n_sims))
    gd = np.zeros((4, n_sims))
    gf = np.zeros((4, n_sims))

    for m in matches:
        a, b = idx[m["home"]], idx[m["away"]]
        sa, sb = strengths[m["home"]], strengths[m["away"]]
        total = sa + sb
        lam_a = normal_goals * sa / total
        lam_b = normal_goals * sb / total
        ga = rng.poisson(lam_a, n_sims)
        gb = rng.poisson(lam_b, n_sims)

        gf[a] += ga
        gf[b] += gb
        gd[a] += ga - gb
        gd[b] += gb - ga
        points[a] += np.where(ga > gb, 3, np.where(ga == gb, 1, 0))
        points[b] += np.where(gb > ga, 3, np.where(gb == ga, 1, 0))

    # Ranking key: points, then goal difference, then goals scored, then a tiny
    # random jitter to break remaining ties uniformly.
    jitter = rng.random((4, n_sims)) * 1e-6
    score = points * 1e6 + gd * 1e3 + gf + jitter
    # order[0] = index of the team ranked 1st in each sim, etc.
    order = np.argsort(-score, axis=0)
    placement = np.zeros((4, 4))
    for p in range(4):
        teams_at_p = order[p]  # shape (n_sims,)
        counts = np.bincount(teams_at_p, minlength=4)
        placement[:, p] = counts / n_sims
    return placement


def simulate_all(
    groups: dict[str, list[dict]],
    group_teams: dict[str, list[str]],
    strengths: dict[str, float],
    normal_goals: float,
    n_sims: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Return {team: placement_probs[4]} across all groups."""
    result: dict[str, np.ndarray] = {}
    for name, matches in groups.items():
        teams = group_teams[name]
        placement = simulate_group(matches, teams, strengths, normal_goals, n_sims, rng)
        for t, team in enumerate(teams):
            result[team] = placement[t]
    return result


def expected_finish(probs: np.ndarray) -> float:
    """Expected finishing position (1..4) from placement probabilities."""
    return float(np.dot(probs, [1, 2, 3, 4]))


def main() -> None:

    params = json.loads(PARAMS_IN.read_text())
    normal_goals = params["normal_goals_per_match"]
    strengths = {team: UNIFORM_START_STRENGTH for team in params["teams"]}

    groups = json.loads(GROUPS_FILE.read_text())
    group_teams = {
        name: sorted({m["home"] for m in matches} | {m["away"] for m in matches})
        for name, matches in groups.items()
    }
    targets = load_targets()

    target_finish = {team: expected_finish(p) for team, p in targets.items()}

    rng = np.random.default_rng(SEED)
    for it in range(N_ITERS):
        # Reset the seed each iteration so the objective is stable across steps.
        rng = np.random.default_rng(SEED)
        sim = simulate_all(groups, group_teams, strengths, normal_goals, N_SIMS, rng)
        max_delta = 0.0
        for team, probs in sim.items():
            e_sim = expected_finish(probs)
            delta = e_sim - target_finish[team]
            strengths[team] *= np.exp(LR * delta)
            max_delta = max(max_delta, abs(delta))
        if (it + 1) % 10 == 0 or it == 0:
            print(f"iter {it + 1:3d}  max |E_sim - E_target| = {max_delta:.4f}")

    # Final, higher-resolution evaluation.
    rng = np.random.default_rng(SEED + 1)
    sim = simulate_all(groups, group_teams, strengths, normal_goals, N_SIMS_FINAL, rng)

    print("\nCalibrated vs target placement probabilities")
    print(f"{'Team':<20}{'sim 1/2/3/4':<28}{'target 1/2/3/4':<28}{'MAE':>7}")
    abs_errors = []
    for team in sorted(sim, key=lambda t: -strengths[t]):
        s = sim[team]
        t = targets[team]
        err = np.abs(s - t)
        abs_errors.extend(err.tolist())
        s_str = "/".join(f"{x:4.0%}" for x in s)
        t_str = "/".join(f"{x:4.0%}" for x in t)
        print(f"{team:<20}{s_str:<28}{t_str:<28}{err.mean():>7.1%}")

    print(f"\nOverall MAE across all team/placement cells: {np.mean(abs_errors):.2%}")

    out = {
        "normal_goals_per_match": normal_goals,
        "teams": {team: round(strengths[team]) for team in params["teams"]},
    }
    PARAMS_OUT.write_text(json.dumps(out, indent=4) + "\n")
    print(f"\nWrote calibrated parameters to {PARAMS_OUT}")


if __name__ == "__main__":
    main()
