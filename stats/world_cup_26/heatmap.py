import collections
import math

import plotly.graph_objects as go

from simulate_match import simulate_match

TEAM1 = "Norge"
TEAM2 = "Brasil"
N = 1000
IS_KNOCKOUT = True

win_a = win_b = draws = penalties = 0
final_score_counts: dict[tuple[int, int], int] = collections.defaultdict(int)

for _ in range(N):
    g_a, g_b, winner, log = simulate_match(TEAM1, TEAM2, is_knockout=IS_KNOCKOUT)
    final_score_counts[(g_a, g_b)] += 1
    went_to_penalties = any(
        "straffer" in line.lower() and "vinner" in line.lower() for line in log
    )
    if went_to_penalties:
        penalties += 1
    elif winner == TEAM1:
        win_a += 1
    elif winner == TEAM2:
        win_b += 1
    else:
        draws += 1

print(f"\n--- {N} simuleringer: {TEAM1} vs {TEAM2} ---")
print(f"  {TEAM1} vinner (reg/ET):  {win_a / N:.1%}")
if IS_KNOCKOUT:
    print(f"  Straffer:              {penalties / N:.1%}")
else:
    print(f"  Uavgjort:              {draws / N:.1%}")
print(f"  {TEAM2} vinner (reg/ET):  {win_b / N:.1%}")

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
            title="Frekvens (%)",
            tickvals=[-max_abs, -max_abs / 2, 0, max_abs / 2, max_abs],
            ticktext=[
                f"−{max_abs:.0f}% ({TEAM2})",
                f"−{max_abs / 2:.0f}%",
                "0% (uavgjort)",
                f"+{max_abs / 2:.0f}%",
                f"+{max_abs:.0f}% ({TEAM1})",
            ],
        ),
    )
)
fig.update_layout(
    title=f"Scoringsfordeling: {TEAM1} vs {TEAM2} ({N} simuleringer)<br>"
    f"<sup style='color:#c0392b'>■ {TEAM1} vinner</sup>  "
    f"<sup style='color:#2d6a4f'>■ {TEAM2} vinner</sup>  "
    f"<sup style='color:#888'>■ Uavgjort/straffer</sup>",
    xaxis_title=f"Mål — {TEAM1}",
    yaxis_title=f"Mål — {TEAM2}",
    xaxis=dict(side="bottom"),
    width=700,
    height=620,
)
fig.show()
