import json
from pathlib import Path

from simulate_match import simulate_match

group_stage: dict[str, list[dict]] = json.loads(
    (Path(__file__).parent / "group_stage.json").read_text()
)

for group_name, matches in group_stage.items():
    print(f"\n{'=' * 45}")
    print(f"  {group_name}")
    print(f"{'=' * 45}")

    for match in matches:
        home = match["home"]
        away = match["away"]
        g_a, g_b, winner, _ = simulate_match(home, away, is_knockout=False)
        print(f"  {home} {g_a}–{g_b} {away}  →  {winner}")

print()
