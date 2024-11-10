"""
Created 02 February 2024
Kristoffer Nesland, kristoffernesland@gmail.com

Inspired by: https://youtu.be/FxrTtRvYQWk?si=dozMG041wJPr0PQG
"""

import torch
import torch.nn as nn


def moe_loss_naive_fn(weights, preds, y):
    """
    Not sure how this would compare to a simple MSELoss on pred and y...
    """
    squared_errors = torch.square(preds - y.reshape(-1, 1))
    weighted_squared_errors = weights * squared_errors
    sum_weighted_square_errors = torch.sum(weighted_squared_errors, dim=1)

    return torch.mean(sum_weighted_square_errors)


def moe_loss_gaussian_fn(weights, preds, y):
    one_over_sqrt_2pi = 0.3989422804
    squared_errors = torch.square(preds - y.reshape(-1, 1))

    expert_probs = one_over_sqrt_2pi * torch.exp(-0.5 * squared_errors)
    weighted_expert_probs = weights * expert_probs

    sum_of_weighted_expert_probs = torch.sum(weighted_expert_probs, dim=1)

    # Adding a small constant, epsilon, to avoid log(0) which results in -inf. Thanks, ChatGPT
    epsilon = 1e-8  # Epsilon can be adjusted based on the precision requirements of your application.
    safe_sum_of_weighted_expert_probs = sum_of_weighted_expert_probs + epsilon

    return torch.mean(-torch.log(safe_sum_of_weighted_expert_probs))


class ExpertModel(nn.Module):
    def __init__(
        self, n_inputs: int, depth: int, width: int, final_bias_init: float = 0.0
    ):
        super().__init__()

        self._input_layer = torch.nn.Linear(n_inputs, width)

        linear_layers = []
        for i in range(depth):
            linear_layers.append(nn.Linear(width, width))
            linear_layers.append(nn.ReLU())
        self._linear_layers = torch.nn.Sequential(*linear_layers)

        self._output_layer = torch.nn.Linear(width, 1)

        self._final_bias = torch.nn.Parameter(torch.Tensor([final_bias_init]))

    def forward(self, x):
        z = self._input_layer(x)
        z = self._linear_layers(z)
        y = self._output_layer(z) + self._final_bias

        return y


class GatingModule(nn.Module):
    def __init__(self, n_inputs: int, n_experts: int, depth: int, width: int):
        super().__init__()

        self._input_layer = torch.nn.Linear(n_inputs, width)

        linear_layers = []
        for i in range(depth - 1):
            linear_layers.append(nn.Linear(width, width))
            linear_layers.append(nn.ReLU())
        linear_layers.append(nn.Linear(width, n_experts))
        self._linear_layers = torch.nn.Sequential(*linear_layers)

        self._softmax_layer = nn.Softmax(dim=1)

    def forward(self, x):
        z = self._input_layer(x)
        z = self._linear_layers(z)
        p = self._softmax_layer(z)

        return p


class MoEModel(nn.Module):
    def __init__(
        self,
        experts: list[ExpertModel],
        gating_module: GatingModule,
        final_bias: float = 0.0,
    ):
        super().__init__()

        self._experts = nn.ModuleList(experts)
        self._gating_module = gating_module

    def expert_output(self, x):
        y_list: list[torch.Tensor] = [expert(x) for expert in self._experts]
        return torch.cat(y_list, dim=1)

    def gating_output(self, x):
        return self._gating_module(x)

    def forward(self, x):
        return torch.sum(self.gating_output(x) * self.expert_output(x), dim=1)


def main():
    n_inputs = 5
    n_rows = 10
    n_experts = 3

    x = torch.randn(n_rows, n_inputs)

    e1 = ExpertModel(n_inputs=n_inputs, depth=3, width=20)
    y1 = e1(x)
    print("x", x)
    print("y1", y1)

    g = GatingModule(n_inputs=n_inputs, n_experts=n_experts, depth=1, width=20)
    p = g(x)
    print("p", p)

    experts = [
        ExpertModel(n_inputs=n_inputs, depth=3, width=20) for i in range(n_experts)
    ]

    model = MoEModel(experts=experts, gating_module=g)
    print(model)
    y = model(x)
    print("y", y)


if __name__ == "__main__":
    main()
