import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(
        self,
        n_input_rows: int,
        n_input_cols: int,
        output_dim: int,
    ):
        assert n_input_rows == 9
        assert n_input_cols == 7
        super(DQN, self).__init__()

        self._conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=8,
            kernel_size=5,
            stride=1,
            padding=1,
        )

        self._fc1 = nn.Linear(280, 128)
        self._fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self._conv1(x))
        x = torch.relu(self._conv2(x))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = torch.relu(self._fc1(x))
        x = self._fc2(x)
        return x


if __name__ == "__main__":
    # Example usage
    n_input_rows = 9
    n_input_cols = 7
    batch_size = 4
    output_dim = 10
    model = DQN(
        n_input_rows=n_input_rows,
        n_input_cols=n_input_cols,
        output_dim=output_dim,
    )
    print(model)
    input_data = torch.randn(batch_size, 1, n_input_rows, n_input_cols)
    output = model(input_data)

    print(output)
