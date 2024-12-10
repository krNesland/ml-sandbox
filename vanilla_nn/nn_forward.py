import numpy as np
import plotly.graph_objects as go
import torch


class Network(torch.nn.Module):
    """
    Output should be relu(x * w00 + y * w01 + b0) + relu(x * w10 + y * w11 + b1)
    """

    def __init__(
        self,
        w00: float,
        w01: float,
        w10: float,
        w11: float,
    ):
        super(Network, self).__init__()

        self._input_size = 2
        self._hidden_size = 2
        self._outputut_size = 1

        self._fc1 = torch.nn.Linear(self._input_size, self._hidden_size)
        self._relu = torch.nn.ReLU()
        self._fc2 = torch.nn.Linear(self._hidden_size, self._outputut_size)

        # Manually set weights and biases
        with torch.no_grad():
            self._fc1.weight = torch.nn.Parameter(
                torch.tensor([[w00, w01], [w10, w11]])
            )
            self._fc1.bias = torch.nn.Parameter(torch.tensor([0.0, 0.0]))

            self._fc2.weight = torch.nn.Parameter(torch.tensor([[1.0, 1.0]]))
            self._fc2.bias = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, xy):
        z = self._fc1(xy)
        z = self._relu(z)
        z = self._fc2(z)

        return z


if __name__ == "__main__":
    net = Network(
        w00=3.0,
        w01=1.0,
        w10=1.0,
        w11=1.0,
    )

    x_inputs = np.linspace(-1.0, 1.0, 100)
    y_inputs = np.linspace(-1.0, 1.0, 100)

    xy_inputs = np.array(np.meshgrid(x_inputs, y_inputs)).T.reshape(-1, 2)

    input_tensor = torch.tensor(xy_inputs, dtype=torch.float32)
    output_tensor = net(input_tensor)
    outputs = output_tensor.detach().numpy()

    fig = go.Figure(
        data=[
            go.Heatmap(
                x=xy_inputs[:, 0],
                y=xy_inputs[:, 1],
                z=outputs[:, 0],
            )
        ]
    )

    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
    )

    fig.show()
