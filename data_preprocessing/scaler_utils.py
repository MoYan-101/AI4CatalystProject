"""
data_preprocessing/scaler_utils.py

Contains functions for data standardization (using StandardScaler),
as well as saving/loading scaler objects for future use.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def standardize_data(X_train, X_val,
                     Y_train, Y_val,
                     do_input=True, do_output=False,
                     numeric_cols_idx=None):
    """
    Optionally standardize input features (X) and/or output targets (Y).
    We skip one-hot columns for X by only applying the scaler to numeric_cols_idx.

    :param X_train: (N_train, input_dim)
    :param X_val:   (N_val,   input_dim)
    :param Y_train: (N_train, output_dim)
    :param Y_val:   (N_val,   output_dim)
    :param do_input: whether to standardize X
    :param do_output: whether to standardize Y
    :param numeric_cols_idx: list of column indices in X that are numeric
    :return:
       (X_train_s, X_val_s, scaler_x), (Y_train_s, Y_val_s, scaler_y)
    """
    scaler_x = None
    scaler_y = None

    # Make copies to avoid modifying original
    X_train_s = np.copy(X_train)
    X_val_s   = np.copy(X_val)
    Y_train_s = np.copy(Y_train)
    Y_val_s   = np.copy(Y_val)

    if do_input:
        if numeric_cols_idx is None:
            # If user didn't provide numeric_cols_idx, we assume all columns are numeric
            numeric_cols_idx = list(range(X_train.shape[1]))

        scaler_x = StandardScaler()
        # Fit only on numeric cols
        scaler_x.fit(X_train_s[:, numeric_cols_idx])

        # Transform
        X_train_s[:, numeric_cols_idx] = scaler_x.transform(X_train_s[:, numeric_cols_idx])
        X_val_s[:, numeric_cols_idx]   = scaler_x.transform(X_val_s[:, numeric_cols_idx])
    # else: no standardization on X

    if do_output:

        Y_train_s = scaler_y.transform(Y_train_s)
        Y_val_s   = scaler_y.transform(Y_val_s)
    # else: no standardization on Y

    return (X_train_s, X_val_s, scaler_x), (Y_train_s, Y_val_s, scaler_y)


def save_scaler(scaler, path):
    """
    Save a fitted scaler to disk using joblib.
    :param scaler: the StandardScaler (or similar) object
    :param path: file path to save
    """
    if scaler is not None:
        joblib.dump(scaler, path)


def load_scaler(path):
    """
    Load a scaler from disk.
    :param path: file path
    :return: loaded scaler
    """
    return joblib.load(path)


def inverse_transform_output(y_pred, scaler_y):
    """
    If output was scaled, invert the transformation for final predictions.
    :param y_pred: predicted outputs, shape (N, output_dim)
    :param scaler_y: the fitted scaler for output
    :return: predictions in original scale
    """
    if scaler_y is None:
        return y_pred
    return scaler_y.inverse_transform(y_pred)
