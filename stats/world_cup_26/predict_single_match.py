"""
Multiple draws for a single match.
"""

import collections
import math

import plotly.graph_objects as go
from simulate_match import simulate_match
from tabulate import tabulate

TEAM1 = "Norway"
TEAM2 = "France"
N = 10_000
IS_KNOCKOUT = False

win_a = win_b = draws = penalties = 0
final_score_counts: dict[tuple[int, int], int] = collections.defaultdict(int)

for _ in range(N):
    g_a, g_b, winner, log = simulate_match(TEAM1, TEAM2, is_knockout=IS_KNOCKOUT)
    final_score_counts[(g_a, g_b)] += 1
    went_to_penalties = any(
        "penalties" in line.lower() and "winner" in line.lower() for line in log
    )
    if went_to_penalties:
        penalties += 1
    elif winner == TEAM1:
        win_a += 1
    elif winner == TEAM2:
        win_b += 1
    else:
        draws += 1

p_a = win_a / N
p_draw = draws / N
p_b = win_b / N
p_pen = penalties / N

odds_a = 1 / p_a if p_a > 0 else float("inf")
odds_draw = 1 / p_draw if p_draw > 0 else float("inf")
odds_b = 1 / p_b if p_b > 0 else float("inf")

rows = [
    [f"{TEAM1} wins", f"{p_a:.1%}", f"{odds_a:.2f}"],
]
if IS_KNOCKOUT:
    rows.append(["Penalties", f"{p_pen:.1%}", "—"])
else:
    rows.append(["Draw", f"{p_draw:.1%}", f"{odds_draw:.2f}"])
rows.append([f"{TEAM2} wins", f"{p_b:.1%}", f"{odds_b:.2f}"])

print(f"\n--- {N} simulations: {TEAM1} vs {TEAM2} ---")
print(
    tabulate(
        rows,
        headers=["Outcome", "Prob", "Odds"],
        tablefmt="simple",
        colalign=("left", "right", "right"),
    )
)

# Build heatmap matrix
# z_color uses sign to encode winner: +pct = team1 wins, -pct = team2 wins, 0 = draw/penalties
max_goals = max(
    max(g for g, _ in final_score_counts), max(g for _, g in final_score_counts), 5
)
grid_size = max_goals + 1
z_color = [[float("nan")] * grid_size for _ in range(grid_size)]
text = [[""] * grid_size for _ in range(grid_size)]

for (ga, gb), count in final_score_counts.items():
    if ga < grid_size and gb < grid_size:
        pct = count / N * 100
        if ga > gb:
            z_color[gb][ga] = pct
        elif gb > ga:
            z_color[gb][ga] = -pct
        else:
            z_color[gb][ga] = 0.0
        text[gb][ga] = f"{pct:.1f}%"

max_abs = max(
    abs(z_color[r][c])
    for r in range(grid_size)
    for c in range(grid_size)
    if not math.isnan(z_color[r][c])
)

colorscale = [
    [0.0, "#2d6a4f"],
    [0.45, "#d8f3dc"],
    [0.5, "#f0f0f0"],
    [0.55, "#ffd6c0"],
    [1.0, "#c0392b"],
]

axis_labels = [str(i) for i in range(grid_size)]
fig = go.Figure(
    go.Heatmap(
        z=z_color,
        x=axis_labels,
        y=axis_labels,
        text=text,
        texttemplate="%{text}",
        colorscale=colorscale,
        zmid=0,
        zmin=-max_abs,
        zmax=max_abs,
        showscale=True,
        colorbar=dict(
            title="Frequency (%)",
            tickvals=[-max_abs, -max_abs / 2, 0, max_abs / 2, max_abs],
            ticktext=[
                f"−{max_abs:.0f}% ({TEAM2})",
                f"−{max_abs / 2:.0f}%",
                "0% (draw)",
                f"+{max_abs / 2:.0f}%",
                f"+{max_abs:.0f}% ({TEAM1})",
            ],
        ),
    )
)
fig.update_layout(
    title=f"Score distribution: {TEAM1} vs {TEAM2} ({N} simulations)<br>"
    f"<sup style='color:#c0392b'>■ {TEAM1} wins</sup>  "
    f"<sup style='color:#2d6a4f'>■ {TEAM2} wins</sup>  "
    f"<sup style='color:#888'>■ Draw/penalties</sup>",
    xaxis_title=f"Goals — {TEAM1}",
    yaxis_title=f"Goals — {TEAM2}",
    xaxis=dict(side="bottom"),
    width=700,
    height=620,
)
fig.show()
