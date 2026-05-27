import json
import math
import random
import urllib.request

CURRENT_ROUND = 6
URL = f"https://fantasy.formula1.com/feeds/drivers/{CURRENT_ROUND}_en.json"

N_RACES_2026 = 22
AVG_RACE_SCORE_AT_100M_BUDGET = 200

# Constants based on 2026 Price Change Algorithm. Note, only valid when there are at least 3 completed rounds.
PPM_TARGET_GREAT = 1.2
PPM_TARGET_GOOD = 0.9
PPM_TARGET_POOR = 0.6
assert CURRENT_ROUND >= 4, (
    "At least 3 completed rounds are required (CURRENT_ROUND is the active feed index). "
    "If not, the PPM target bands are not valid."
)


def _race_sample_weights(n: int) -> list[float]:
    """Linear weights 1..n, e.g. three races → [1/6, 2/6, 3/6]."""
    denom = n * (n + 1) / 2
    return [i / denom for i in range(1, n + 1)]


def _pts_to_great(price: float, p_n2: float, p_n1: float) -> int:
    """Next-round points for 3-race avg PPM > great (> 1.2). Negative = headroom."""
    need = 3 * PPM_TARGET_GREAT * price - p_n2 - p_n1
    if need < 0:
        return math.ceil(need)
    return math.floor(need) + 1


def _simulate_pts(race_points: list[float]) -> float:
    if not race_points:
        return 0.0
    if len(race_points) == 1:
        return race_points[0]
    return random.choices(
        race_points, weights=_race_sample_weights(len(race_points)), k=1
    )[0]


def _fetch_feed(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception:
        return {}


def _get_overall_points(data: dict) -> dict:
    out = {}
    for p in data.get("Data", {}).get("Value", []):
        pid = p.get("PlayerId")
        raw = p.get("OverallPpints", 0)  # Typo in 2026 feed
        out[pid] = float(raw) if raw else 0.0
    return out


def get_2026_budget_strategy():
    data = _fetch_feed(URL)
    if not data:
        return

    n_completed_races = CURRENT_ROUND - 1  # feed index is the active/upcoming round
    n_remaining_races = N_RACES_2026 - n_completed_races
    avg_ppm = AVG_RACE_SCORE_AT_100M_BUDGET / 100
    one_m_worth = avg_ppm * n_remaining_races

    print(
        f"One million dollars is worth {one_m_worth:.2f} points at the end of the season."
    )

    totals_list = []
    for round in range(0, CURRENT_ROUND + 1):
        url = URL.replace(f"/{CURRENT_ROUND}_", f"/{round}_")
        totals = _get_overall_points(_fetch_feed(url))
        totals_list.append(totals)

    all_assets = data.get("Data", {}).get("Value", [])
    results = []

    for p in all_assets:
        pid = p.get("PlayerId")
        name = p.get("FUllName") or p.get("Name")
        price = float(p.get("Value", 0))
        if price == 0:
            continue

        # totals_list[-1] is the ongoing/upcoming round; race k uses totals_list[k] − totals_list[k−1]
        race_pts: list[float] = []
        for i in range(1, len(totals_list) - 1):
            race_pts.append(
                totals_list[i].get(pid, 0.0) - totals_list[i - 1].get(pid, 0.0)
            )

        p_n1 = race_pts[-1] if race_pts else 0.0
        p_n2 = race_pts[-2] if len(race_pts) >= 2 else 0.0

        season_pts = totals_list[-2].get(pid, 0.0)

        pts_to_great = _pts_to_great(price, p_n2, p_n1)

        sim_pts_results = []
        for _ in range(100):
            sim_pts_results.append(_simulate_pts(race_pts))

        exp_pts = sum(sim_pts_results) / len(sim_pts_results)

        results.append(
            {
                "name": name,
                "price": price,
                "season_pts": season_pts,
                "race_pts": race_pts,
                "pts_to_great": pts_to_great,
                "exp_pts": exp_pts,
                "surplus": exp_pts - pts_to_great,
            }
        )

    results.sort(key=lambda x: x["surplus"], reverse=True)

    print("\nColumn guide:")
    print("  2026 ASSET   — Driver or constructor name.")
    print("  PRICE        — Current fantasy value ($M).")
    print(
        "  SEASON       — Total fantasy points of the *completed* rounds this season."
    )
    n_completed = len(results[0]["race_pts"]) if results else 0
    print(
        f"  P_R1…P_R{n_completed} — Points per completed race (R1 = first, R{n_completed} = latest)."
    )
    print("  EXP PTS      — Mean simulated next-round points (sampled from all races;")
    print(
        f"                 linear weights 1…{n_completed} → "
        f"{', '.join(f'{w:.0%}' for w in _race_sample_weights(n_completed)) if n_completed else 'n/a'})."
    )
    print("  TO GREAT     — Points needed in the next round so the 3-race average PPM")
    print("                 (last two races + next) exceeds the great band (> 1.2).")
    print("                 Negative = already great with points to spare.")
    print(
        "  SURPLUS      — EXP PTS − TO GREAT (expected cushion above the great band)."
    )
    print("                 Table is sorted by SURPLUS (highest first).")
    print()

    race_cols = " | ".join(f"{f'P_R{i}':<8}" for i in range(1, n_completed + 1))
    print(
        f"{'2026 ASSET':<22} | {'PRICE':<7} | {'SEASON':<8} | {race_cols} | {'EXP PTS':<8} | "
        f"{'TO GREAT':<8} | {'SURPLUS':<8}"
    )
    print("-" * (88 + 11 * max(0, n_completed - 1)))
    for r in results:
        race_vals = " | ".join(f"{pts:>8.0f}" for pts in r["race_pts"])
        print(
            f"{r['name']:<22} | ${r['price']:>5.1f}M | {r['season_pts']:>8.0f} | "
            f"{race_vals} | {r['exp_pts']:>8.1f} | {r['pts_to_great']:>+8.0f} | {r['surplus']:>+8.1f}"
        )


if __name__ == "__main__":
    get_2026_budget_strategy()
