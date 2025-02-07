"""
models/model_ann.py

Implementation of a simple feedforward ANN for multi-output regression
using PyTorch.
"""

import torch
import torch.nn as nn

class ANNRegression(nn.Module):
    """
    A simple ANN for regression, output_dim=3 by default.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 32]):
        """
        :param input_dim: number of input features
        :param output_dim: number of regression outputs
        :param hidden_dims: list of hidden layer sizes (e.g. [64, 32])
        """
        super().__init__()
        layers = []
        in_dim = input_dim

        # Build hidden layers
        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, hd))
            layers.append(nn.ReLU())
            in_dim = hd

        # Final output layer
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the feedforward network.
        :param x: Tensor shape (batch_size, input_dim)
        :return:  Tensor shape (batch_size, output_dim)
        """
        return self.net(x)
