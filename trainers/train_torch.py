"""
trainer/train_torch.py

Provides a training loop for PyTorch-based models using DataLoader,
plus best-checkpoint saving.
"""

import torch
import numpy as np
import copy
from torch.utils.data import DataLoader

def train_torch_model_dataloader(model,
                                 train_dataset,
                                 val_dataset,
                                 loss_fn,
                                 epochs=30,
                                 batch_size=32,
                                 lr=1e-3,
                                 checkpoint_path=None,
                                 log_interval=5):
    """
    Train a PyTorch model using DataLoader for the dataset.
    :param model: nn.Module, the model to train
    :param train_dataset: PyTorch Dataset for training
    :param val_dataset: PyTorch Dataset for validation
    :param loss_fn: callable that computes the loss (e.g., MSE)
    :param epochs: total number of training epochs
    :param batch_size: mini-batch size
    :param lr: learning rate for optimizer
    :param checkpoint_path: if provided, best model weights will be saved here
    :param log_interval: how often to log epoch stats
    :return: (model, train_losses, val_losses)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs+1):
        # -- Training --
        model.train()
        batch_loss_list = []
        for (X_batch, Y_batch) in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # Forward
            pred_batch = model(X_batch)
            loss_batch = loss_fn(pred_batch, Y_batch)

            # Backward
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            batch_loss_list.append(loss_batch.item())

        train_loss = np.mean(batch_loss_list)

        # -- Validation --
        model.eval()
        val_loss_list = []
        with torch.no_grad():
            for (X_val_b, Y_val_b) in val_loader:
                X_val_b = X_val_b.to(device)
                Y_val_b = Y_val_b.to(device)

                val_pred_b = model(X_val_b)
                loss_b = loss_fn(val_pred_b, Y_val_b)
                val_loss_list.append(loss_b.item())

        val_loss = np.mean(val_loss_list)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % log_interval == 0:
            print(f"[Epoch {epoch}/{epochs}] "
                  f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)

    # After training, restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model with val_loss={best_val_loss:.6f}")

    return model, train_losses, val_losses
