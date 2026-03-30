import json
import random
import re
import urllib.request

CURRENT_ROUND = 4
URL = f"https://fantasy.formula1.com/feeds/drivers/{CURRENT_ROUND}_en.json"

N_RACES_2026 = 22
AVG_RACE_SCORE_AT_100M_BUDGET = 200

# Constants based on 2026 Price Change Algorithm
PPM_TARGET_GREAT = 1.2
PPM_TARGET_GOOD = 0.9
PPM_TARGET_POOR = 0.6
TIER_A_THRESHOLD = 18.5


def _get_expected_price_change(pred_ppm: float, tier_a: bool) -> float:
    if pred_ppm >= PPM_TARGET_GREAT:
        return 0.3 if tier_a else 0.6
    elif pred_ppm >= PPM_TARGET_GOOD:
        return 0.1 if tier_a else 0.2
    elif pred_ppm >= PPM_TARGET_POOR:
        return -0.1 if tier_a else -0.2
    else:
        return -0.3 if tier_a else -0.6


def _simulate_pts(p1: float, p2: float) -> float:
    return random.sample([p1, p2], 1)[0]


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

    n_remaining_races = N_RACES_2026 - CURRENT_ROUND
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

        p_n1: float  # Points in the last *completed* round
        p_n2: float  # Points in the second-to-last *completed* round

        # totals_list[-1] is the ongoing/upcoming round; completed rounds use -2, -3, -4…
        if len(totals_list) >= 4:
            p_n1 = totals_list[-2].get(pid, 0.0) - totals_list[-3].get(pid, 0.0)
            p_n2 = totals_list[-3].get(pid, 0.0) - totals_list[-4].get(pid, 0.0)
        elif len(totals_list) >= 3:
            p_n1 = totals_list[-2].get(pid, 0.0) - totals_list[-3].get(pid, 0.0)
            p_n2 = 0.0
        else:
            p_n1 = 0.0
            p_n2 = 0.0

        season_pts = totals_list[-2].get(pid, 0.0)

        is_tier_a = price >= TIER_A_THRESHOLD

        sim_change_results = []
        sim_ppm_results = []
        for _ in range(100):
            sim_pts = _simulate_pts(p_n2, p_n1)
            sim_ppm = (p_n2 + p_n1 + sim_pts) / price
            sim_exp_change = _get_expected_price_change(sim_ppm, is_tier_a)
            sim_change_results.append(sim_exp_change)
            sim_ppm_results.append(sim_ppm)

        exp_change = sum(sim_change_results) / len(sim_change_results)
        exp_ppm = sum(sim_ppm_results) / len(sim_ppm_results)

        results.append(
            {
                "name": name,
                "price": price,
                "season_pts": season_pts,
                "p_n2": p_n2,
                "p_n1": p_n1,
                "exp_ppm": exp_ppm,
                "exp_change": exp_change,
            }
        )

    results.sort(key=lambda x: x["exp_ppm"], reverse=True)

    print("\nColumn guide:")
    print("  2026 ASSET   — Driver or constructor name.")
    print("  PRICE        — Current fantasy value ($M).")
    print(
        "  SEASON       — Total fantasy points of the *completed* rounds this season."
    )
    print("  P_N2         — Points in the second-to-last *completed* round.")
    print("  P_N1         — Points in the last *completed* round.")
    print("  EXP PPM      — Mean simulated expected PPM.")
    print(
        "  EXP CHANGE   — Mean simulated expected price move ($M) from the 2026 PPM bands."
    )
    print("                 Table is sorted by this column (highest first).")
    print(
        "  PTS EQUIV    — Estimated fantasy points that move is worth over the remaining"
    )
    print(
        "                 races (EXP CHANGE × the “one million dollars” points rate above)."
    )
    print()

    print(
        f"{'2026 ASSET':<22} | {'PRICE':<7} | {'SEASON':<8} | {'P_N2':<8} | {'P_N1':<8} | {'EXP PPM':<8} | "
        f"{'EXP CHANGE':<11} | {'PTS EQUIV':>10}"
    )
    print("-" * 103)
    for r in results:
        change_str = (
            f"{'+' if r['exp_change'] >= 0 else '-'}${abs(r['exp_change']):.2f}M"
        )
        pts_equiv = r["exp_change"] * one_m_worth
        pts_str = f"{pts_equiv:+10.1f}"
        print(
            f"{r['name']:<22} | ${r['price']:>5.1f}M | {r['season_pts']:>8.0f} | "
            f"{r['p_n2']:>8.0f} | {r['p_n1']:>8.0f} | {r['exp_ppm']:>8.2f} | {change_str:<11} | {pts_str}"
        )


if __name__ == "__main__":
    get_2026_budget_strategy()
