"""
Created 04 February 2024
Kristoffer Nesland, kristoffernesland@gmail.com
"""

import torch
import torch.nn as nn

from moe.model import ExpertModel, GatingModule, MoEModel, moe_loss_naive_fn


def main():
    n_inputs = 3
    n_rows = 20
    n_experts = 5

    x = torch.randn(n_rows, n_inputs)
    y = torch.randn(n_rows)
    g = GatingModule(n_inputs=n_inputs, n_experts=n_experts, depth=1, width=20)
    experts = [
        ExpertModel(n_inputs=n_inputs, depth=3, width=20) for i in range(n_experts)
    ]
    model = MoEModel(experts=experts, gating_module=g)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    for i in range(10_000):
        weights = model.gating_output(x)
        preds = model.expert_output(x)

        loss = moe_loss_naive_fn(weights, preds, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        print(f"loss: {loss:>7f}")

    with torch.no_grad():
        pred = model(x)

        print(pred)
        print(y)


if __name__ == "__main__":
    main()
