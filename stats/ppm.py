import json
import math
import random
import urllib.request

CURRENT_ROUND = 6
URL = f"https://fantasy.formula1.com/feeds/drivers/{CURRENT_ROUND}_en.json"
SCHEDULE_URL = "https://fantasy.formula1.com/feeds/v2/schedule/raceday_en.json"

N_RACES_2026 = 22
AVG_RACE_SCORE_AT_100M_BUDGET = 200

# Constants based on 2026 Price Change Algorithm. Note, only valid when there are at least 3 completed rounds.
PPM_TARGET_GREAT = 1.2
PPM_TARGET_GOOD = 0.9
PPM_TARGET_POOR = 0.6
TIER_A_THRESHOLD = 18.5
assert CURRENT_ROUND >= 4, (
    "At least 3 completed rounds are required (CURRENT_ROUND is the active feed index). "
    "If not, the PPM target bands are not valid."
)


def _race_sample_weights(n: int) -> list[float]:
    """Linear weights 1..n, e.g. three races → [1/6, 2/6, 3/6]."""
    denom = n * (n + 1) / 2
    return [i / denom for i in range(1, n + 1)]


def _price_change_from_ppm(avg_ppm: float, tier_a: bool) -> float:
    if avg_ppm >= PPM_TARGET_GREAT:
        return 0.3 if tier_a else 0.6
    if avg_ppm >= PPM_TARGET_GOOD:
        return 0.1 if tier_a else 0.2
    if avg_ppm >= PPM_TARGET_POOR:
        return -0.1 if tier_a else -0.2
    return -0.3 if tier_a else -0.6


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
        race_points,
        weights=_race_sample_weights(len(race_points)),
        k=1,
    )[0]


def _fetch_feed(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception:
        return {}


def _get_gameday_points(data: dict) -> dict:
    """Per-round fantasy points from the round's feed (not cumulative OverallPpints)."""
    out = {}
    for p in data.get("Data", {}).get("Value", []):
        pid = p.get("PlayerId")
        raw = p.get("GamedayPoints", 0)
        out[pid] = float(raw) if raw else 0.0
    return out


def _get_race_venues() -> list[str]:
    """Round number → venue from the fantasy schedule feed."""
    circuits = _fetch_feed(SCHEDULE_URL).get("Data", {}).get("circuit", [])
    by_round = {
        c["MeetingNumber"]: c.get("CircuitLocation") or c.get("MeetingName", "?")
        for c in circuits
        if c.get("MeetingNumber") is not None
    }
    if not by_round:
        return []
    return [by_round.get(i, "?") for i in range(1, max(by_round) + 1)]


def get_2026_budget_strategy():
    data = _fetch_feed(URL)
    if not data:
        return

    race_venues = _get_race_venues()
    n_season_races = len(race_venues) or N_RACES_2026
    n_completed_races = CURRENT_ROUND - 1  # feed index is the active/upcoming round
    n_remaining_races = n_season_races - n_completed_races
    avg_ppm = AVG_RACE_SCORE_AT_100M_BUDGET / 100
    one_m_worth = avg_ppm * n_remaining_races

    print(
        f"One million dollars is worth {one_m_worth:.2f} points at the end of the season."
    )

    race_feeds = [
        _fetch_feed(URL.replace(f"/{CURRENT_ROUND}_", f"/{round}_"))
        for round in range(1, CURRENT_ROUND)
    ]

    all_assets = data.get("Data", {}).get("Value", [])
    results = []

    for p in all_assets:
        pid = p.get("PlayerId")
        name = p.get("FUllName") or p.get("Name")
        price = float(p.get("Value", 0))
        if price == 0:
            continue

        race_pts = [
            _get_gameday_points(f).get(pid, 0.0) for f in race_feeds
        ]

        p_n1 = race_pts[-1] if race_pts else 0.0
        p_n2 = race_pts[-2] if len(race_pts) >= 2 else 0.0

        season_pts = sum(race_pts)

        pts_to_great = _pts_to_great(price, p_n2, p_n1)
        tier_a = price >= TIER_A_THRESHOLD

        sim_pts_results: list[float] = []
        sim_change_results: list[float] = []
        for _ in range(100):
            sim_pts = _simulate_pts(race_pts)
            sim_ppm = (p_n2 + p_n1 + sim_pts) / 3 / price
            sim_change_results.append(_price_change_from_ppm(sim_ppm, tier_a))
            sim_pts_results.append(sim_pts)

        exp_pts = sum(sim_pts_results) / len(sim_pts_results)
        exp_change = sum(sim_change_results) / len(sim_change_results)

        results.append(
            {
                "name": name,
                "price": price,
                "season_pts": season_pts,
                "race_pts": race_pts,
                "pts_to_great": pts_to_great,
                "exp_pts": exp_pts,
                "exp_change": exp_change,
            }
        )

    results.sort(key=lambda x: x["exp_change"], reverse=True)

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
    for i in range(n_completed):
        venue = race_venues[i] if i < len(race_venues) else "?"
        print(f"                 R{i + 1} — {venue}")
    print("  EXP PTS      — Mean simulated next-round points (sampled from all races;")
    print(
        f"                 linear weights 1…{n_completed} → "
        f"{', '.join(f'{w:.0%}' for w in _race_sample_weights(n_completed)) if n_completed else 'n/a'})."
    )
    print("  TO GREAT     — Points needed in the next round so the 3-race average PPM")
    print("                 (last two races + next) exceeds the great band (> 1.2).")
    print("                 Negative = already great with points to spare.")
    print("  EXP CHANGE   — Mean simulated price move ($M) from the 2026 PPM bands")
    print("                 (avg of last two races + sampled next round).")
    print("                 Table is sorted by EXP CHANGE (highest first).")
    print()

    race_cols = " | ".join(f"{f'P_R{i}':<8}" for i in range(1, n_completed + 1))
    print(
        f"{'2026 ASSET':<22} | {'PRICE':<7} | {'SEASON':<8} | {race_cols} | {'EXP PTS':<8} | "
        f"{'TO GREAT':<8} | {'EXP CHANGE':<11}"
    )
    print("-" * (99 + 11 * max(0, n_completed - 1)))
    for r in results:
        race_vals = " | ".join(f"{pts:>8.0f}" for pts in r["race_pts"])
        sign = "+" if r["exp_change"] >= 0 else "-"
        change_str = f"{sign}${abs(r['exp_change']):.2f}M"
        print(
            f"{r['name']:<22} | ${r['price']:>5.1f}M | {r['season_pts']:>8.0f} | "
            f"{race_vals} | {r['exp_pts']:>8.1f} | {r['pts_to_great']:>+8.0f} | {change_str:<11}"
        )


if __name__ == "__main__":
    get_2026_budget_strategy()
    # TODO: Add some cross-learning? When sampling points, sample not only based on that specific asset, but also based on the other assets in the portfolio.
