"""
Created 05 February 2024
Kristoffer Nesland, kristoffernesland@gmail.com

What is the mean error if we estimate the mean of the y?
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from moe.data_utils import get_eval_data, get_train_data


def main():
    train_x, train_y = get_train_data()

    mean_y = torch.mean(train_y)
    pred = mean_y
    print(pred.item())

    errors = torch.square(train_y - pred)
    print(errors)
    print(torch.mean(errors).item())


if __name__ == "__main__":
    main()
