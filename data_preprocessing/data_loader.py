"""
data_preprocessing/data_loader.py

Revised version to explicitly:
1) Only take first 14 columns from CSV (0..13)
2) Drop or fill NaNs in those columns
3) Then parse columns 0..9 => X_df, columns 10..13 => Y
4) Perform one-hot on X_df
5) Return (X, Y, numeric_cols_idx)
"""

import pandas as pd
import numpy as np

def load_dataset(csv_path, drop_nan=True):
    """
    Read a CSV file, forcibly keep only the first 14 columns.
    Then parse the first 10 as X, the next 4 as Y.
    Perform one-hot encoding for categorical features in X.
    Finally, return X (NumPy), Y (NumPy), numeric_cols_idx (for standardization).

    :param csv_path: Path to the CSV file .
    :param drop_nan: if True, drop any row that has NaN in first 14 columns
                     if False, you could fill them or handle differently
    :return: (X, Y, numeric_cols)
      - X: shape (N, new_dim) after one-hot
      - Y: shape (N, 4)
      - numeric_cols_idx: list of column indices in X that correspond to numeric (non-onehot) features
    """

    # 1) 读取CSV, 强制只保留前14列
    df_raw = pd.read_csv(csv_path)
    # 截取前14列（索引0..13）
    df = df_raw.iloc[:, :14].copy()

    # 2) 处理空值: 选择drop或fill
    if drop_nan:
        # Drop rows that have NaN in these 14 columns
        df.dropna(subset=df.columns, how='any', inplace=True)
    else:
        # 或者你可以做填充, 例如 fillna(0)
        # df.fillna(0, inplace=True)
        pass

    # 再检查一下，是否仍有NaN
    # print("After dropping/cleaning, any NaN?\n", df.isna().sum())

    # 3) 拆分 X_df, Y
    #   - 前10列 => X_df
    #   - 后4列 => Y ( shape (N, 4) )
    if df.shape[1] < 13:
        raise ValueError("After dropping NaNs, not enough columns remain (need 14). Check your CSV or data cleaning logic.")

    X_df = df.iloc[:, :10].copy()
    Y_df = df.iloc[:, 10:14].copy()  # shape: (N,4)

    y_col_names = list(Y_df.columns)  # 后4列列名

    # 转成 NumPy
    Y = Y_df.values  # shape (N,4)

    # 4) 检测哪些列是categorical
    categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()

    # 确定 numeric 列(原始、未OneHot时)
    numeric_cols_original = [
        col for col in X_df.columns
        if col not in categorical_cols
    ]

    # 5) One-hot 对 categorical
    X_encoded = pd.get_dummies(X_df, columns=categorical_cols)

    # 6) 找到 numeric_cols_idx 在 one-hot 后的X_encoded中的位置
    all_cols = X_encoded.columns.tolist()
    numeric_cols_idx = []
    for i, colname in enumerate(all_cols):
        if colname in numeric_cols_original:
            numeric_cols_idx.append(i)

    # 转成 NumPy array
    X = X_encoded.values  # shape: (N, new_num_dim)
    x_col_names = list(X_encoded.columns)  # 记录 one-hot 后的列名

    return X, Y, numeric_cols_idx, x_col_names, y_col_names


def load_raw_data_for_correlation(csv_path, drop_nan=True):
    """
    如果要做混合变量相关性分析，而不想做 One-Hot，可用这个读取前14列。
    """
    df_raw = pd.read_csv(csv_path)
    df = df_raw.iloc[:, :14].copy()

    if drop_nan:
        df.dropna(subset=df.columns, how='any', inplace=True)

    return df
