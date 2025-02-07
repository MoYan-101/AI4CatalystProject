"""
evaluation/metrics.py

Implements common regression metrics: MSE, MAE, R2.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_regression_metrics(y_true, y_pred):
    """
    Compute MSE, MAE, and R2 for the given true/predicted values.
    :param y_true: shape (N, output_dim)
    :param y_pred: shape (N, output_dim)
    :return: a dict with "MSE", "MAE", "R2"
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred, multioutput='uniform_average')
    return {
        "MSE": mse,
        "MAE": mae,
        "R2" : r2
    }


if __name__ == "__main__":
    # 假设真实值和预测值
    y_true = np.array([[3.5, 2.1], [4.0, 3.3], [5.2, 6.8]])
    y_pred = np.array([[3.7, 2.0], [4.1, 3.5], [5.0, 6.5]])

    # 计算回归指标
    metrics = compute_regression_metrics(y_true, y_pred)
    print(metrics)