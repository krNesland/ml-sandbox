import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


if __name__ == "__main__":
    # Example usage
    input_dim = 10
    output_dim = 5

    model = DQN(input_dim, output_dim)
    input_data = torch.randn(1, input_dim)
    output = model(input_data)

    print(output)