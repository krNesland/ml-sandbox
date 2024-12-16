import plotly.graph_objects as go


def animation_from_figures(figures: list[go.Figure]) -> go.Figure:
    # Create frames from the figures
    frames = [go.Frame(data=fig.data, name=str(i)) for i, fig in enumerate(figures)]

    # Create the initial figure
    initial_fig = figures[0]

    # Create the animated figure
    animated_fig = go.Figure(
        data=initial_fig.data, layout=initial_fig.layout, frames=frames
    )

    # Update the layout to include animation controls
    animated_fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ]
    )

    return animated_fig.show()
