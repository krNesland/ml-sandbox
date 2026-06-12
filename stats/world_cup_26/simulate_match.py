# Gemini reverse-engineered the parameters from: https://www.vg.no/spesial/2026/fotball-vm/sjansene/


import json
import math
import random
from pathlib import Path

_data = json.loads((Path(__file__).parent / "parameters.json").read_text())
TEAMS: dict[str, int] = _data["teams"]
NORMAL_GOALS: float = _data["normal_goals_per_match"]


def poisson_sample(lam: float) -> int:
    """Generates a random event count using a Poisson distribution."""
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1


def simulate_match(
    team_a: str, team_b: str, is_knockout: bool = False
) -> tuple[int, int, str, list[str]]:
    """
    Simulates a single match between team_a and team_b.

    Returns:
        (goals_a, goals_b, winner, match_log)

    winner is the team name, "Draw" for a draw (group stage),
    or the penalty winner if the match went to penalties.
    match_log contains a human-readable play-by-play.
    """
    if team_a not in TEAMS or team_b not in TEAMS:
        raise ValueError(f"Unknown team: '{team_a}' or '{team_b}'.")

    strength_a = TEAMS[team_a]
    strength_b = TEAMS[team_b]

    total_strength = strength_a + strength_b
    lambda_a = NORMAL_GOALS * (strength_a / total_strength)
    lambda_b = NORMAL_GOALS * (strength_b / total_strength)

    goals_a = poisson_sample(lambda_a)
    goals_b = poisson_sample(lambda_b)

    log = [f"Regular time (90 min): {team_a} {goals_a} – {goals_b} {team_b}"]

    if goals_a == goals_b and is_knockout:
        log.append("Extra time (30 min)...")
        goals_a += poisson_sample(lambda_a * (30 / 90))
        goals_b += poisson_sample(lambda_b * (30 / 90))
        log.append(f"After extra time: {team_a} {goals_a} – {goals_b} {team_b}")

        if goals_a == goals_b:
            log.append("Penalty shootout...")
            penalty_winner = random.choice([team_a, team_b])
            log.append(f"Winner on penalties: {penalty_winner}!")
            return goals_a, goals_b, penalty_winner, log

    if goals_a > goals_b:
        winner = team_a
    elif goals_b > goals_a:
        winner = team_b
    else:
        winner = "Draw"

    return goals_a, goals_b, winner, log
