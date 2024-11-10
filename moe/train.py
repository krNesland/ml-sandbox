"""
Created 02 February 2024
Kristoffer Nesland, kristoffernesland@gmail.com

TODO: regularization?
"""

import typing as ty
from moe.data_utils import get_eval_data, get_train_data


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from moe.model import (
    ExpertModel,
    GatingModule,
    MoEModel,
    moe_loss_gaussian_fn,
    moe_loss_naive_fn,
)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

torch.manual_seed(0)


def train(dataloader, model, optimizer, epoch: int):
    losses = []
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        weights = model.gating_output(X)
        preds = model.expert_output(X)

        loss = moe_loss_gaussian_fn(weights, preds, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    print("Average loss: ", sum(losses) / len(losses))


def eval_(
    dataloader,
    model,
    epoch: int,
    prefix: str = "eval",
    eval_x_selected: ty.Optional[torch.Tensor] = None,
):
    num_batches = len(dataloader)
    model.eval()
    mae = 0.0
    mse = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            mae += torch.mean(torch.abs(pred - y)).item()
            mse += torch.mean(torch.square(pred - y)).item()

        if eval_x_selected is not None:
            eval_x_selected = eval_x_selected.to(device)

            gating_of_selected = model.gating_output(eval_x_selected).cpu().numpy()
            expert_pred_of_selected = model.expert_output(eval_x_selected).cpu().numpy()

            print(gating_of_selected)
            print(expert_pred_of_selected)

    final_mae = mae / num_batches
    final_mse = mse / num_batches
    print(f"{prefix} error: \n MAE: {final_mae:>8f} \n MSE: {final_mse:>8f} \n")

    if eval_x_selected is not None:
        n_rows = gating_of_selected.shape[0]
        n_experts = gating_of_selected.shape[1]
        for i in range(n_rows):
            for j in range(n_experts):
                print(f"expert_{j}/pred_{i}", expert_pred_of_selected[i, j])
                print(f"gating_{j}/weight_{i}", gating_of_selected[i, j])


def main():
    eval_steps = 10

    gating_depth = 1
    gating_width = 10

    n_experts = 10
    expert_depth = 3
    expert_width = 50

    n_epochs = 10_000
    batch_size = 4096

    train_x, train_y = get_train_data()
    mean_train_y = float(train_y.mean().numpy())
    n_inputs = train_x.shape[1]
    train_dataloader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size
    )

    eval_x, eval_y = get_eval_data()
    eval_dataloader = DataLoader(TensorDataset(eval_x, eval_y), batch_size=batch_size)

    eval_x_selected = eval_x[[1, 3]]

    g = GatingModule(
        n_inputs=n_inputs, n_experts=n_experts, depth=gating_depth, width=gating_width
    )
    experts = [
        ExpertModel(
            n_inputs=n_inputs,
            depth=expert_depth,
            width=expert_width,
            final_bias_init=mean_train_y,
        )
        for i in range(n_experts)
    ]
    model = MoEModel(experts=experts, gating_module=g).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for t in range(n_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(dataloader=train_dataloader, model=model, optimizer=optimizer, epoch=t)

        if (t % eval_steps) == 0:
            eval_(
                dataloader=eval_dataloader,
                model=model,
                epoch=t,
                prefix="eval",
                eval_x_selected=eval_x_selected,
            )
            eval_(
                dataloader=train_dataloader,
                model=model,
                epoch=t,
                prefix="train",
            )


if __name__ == "__main__":
    main()
