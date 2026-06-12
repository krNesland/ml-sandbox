"""
Odds for every match in the group stage.
"""

import json
from pathlib import Path

from simulate_match import simulate_match
from tabulate import tabulate

N = 10_000

group_stage: dict[str, list[dict]] = json.loads(
    (Path(__file__).parent / "group_stage.json").read_text()
)


def match_odds(home: str, away: str, n: int) -> tuple[float, float, float, float, float, float]:
    win_home = win_away = draws = 0

    for _ in range(n):
        _, _, winner, _ = simulate_match(home, away, is_knockout=False)
        if winner == home:
            win_home += 1
        elif winner == away:
            win_away += 1
        else:
            draws += 1

    p_home = win_home / n
    p_draw = draws / n
    p_away = win_away / n

    odds_home = 1 / p_home if p_home > 0 else float("inf")
    odds_draw = 1 / p_draw if p_draw > 0 else float("inf")
    odds_away = 1 / p_away if p_away > 0 else float("inf")

    return p_home, p_draw, p_away, odds_home, odds_draw, odds_away


for group_name, matches in group_stage.items():
    rows = []
    for match in matches:
        home = match["home"]
        away = match["away"]
        p_home, p_draw, p_away, odds_home, odds_draw, odds_away = match_odds(
            home, away, N
        )
        rows.append(
            [
                f"{home} vs {away}",
                f"{p_home:.1%}",
                f"{p_draw:.1%}",
                f"{p_away:.1%}",
                f"{odds_home:.2f}",
                f"{odds_draw:.2f}",
                f"{odds_away:.2f}",
            ]
        )

    print(f"\n--- {group_name} ({N} simulations per match) ---")
    print(
        tabulate(
            rows,
            headers=[
                "Match",
                "Home win",
                "Draw",
                "Away win",
                "Odds H",
                "Odds D",
                "Odds A",
            ],
            tablefmt="simple",
            colalign=("left", "right", "right", "right", "right", "right", "right"),
        )
    )

print()
