 

现在我给出你我的所有的目前的代码,.
├── catboost_info
│   ├── catboost_training.json
│   ├── learn
│   │   └── events.out.tfevents
│   ├── learn_error.tsv
│   ├── time_left.tsv
│   └── tmp
├── check.md
├── configs
│   └── config.yaml
├── data
│   ├── cleansing.ipynb
│   ├── Dataset_20250203.csv
│   ├── Dataset_20250203_upd.csv
│   ├── Dataset_20250203_upd_without1.csv
│   ├── Dataset_20250203_upd.xlsx
│   ├── Dataset_20250203.xlsx
│   ├── Dataset_20250205.csv
│   ├── Dataset_20250205_final.csv
│   ├── Dataset_20250205_without1.csv
│   ├── Dataset_20250210_Cat.csv
│   ├── Dataset_20250210_Cat.xlsx
│   ├── Dataset_20250210_log.csv
│   ├── Dataset_20250210_log.xlsx
│   ├── dataset.csv
│   ├── jupyter.log
│   └── pycharm.log
├── data_preprocessing
│   ├── data_loader.py
│   ├── data_split.py
│   ├── __init__.py
│   ├── my_dataset.py
│   ├── __pycache__
│   │   ├── data_loader.cpython-38.pyc
│   │   ├── data_split.cpython-38.pyc
│   │   ├── __init__.cpython-38.pyc
│   │   ├── my_dataset.cpython-38.pyc
│   │   └── scaler_utils.cpython-38.pyc
│   └── scaler_utils.py
├── Dataset_20250205_final
│   ├── DataCorrelation
│   └── model_comparison
│       ├── RF
│       └── XGB
├── evaluation
│   ├── figures
│   │   ├── Dataset_20250205_final
│   │   │   ├── DataCorrelation
│   │   │   │   ├── catalyst_size_vs_product.jpg
│   │   │   │   ├── correlation_heatmap.jpg
│   │   │   │   ├── correlation_heatmap_one_hot.jpg
│   │   │   │   ├── kde_distribution.jpg
│   │   │   │   ├── overfitting_single.jpg
│   │   │   │   ├── potential_vs_product_by_electrolyte.jpg
│   │   │   │   ├── product_distribution.jpg
│   │   │   │   ├── three_metrics_horizontal_train.jpg
│   │   │   │   └── three_metrics_horizontal_val.jpg
│   │   │   └── model_comparison
│   │   │       ├── ANN
│   │   │       │   ├── ANN_loss_curve.jpg
│   │   │       │   └── full
│   │   │       │       ├── train
│   │   │       │       │   ├── ANN_mae_scatter_train.jpg
│   │   │       │       │   ├── ANN_mse_scatter_train.jpg
│   │   │       │       │   ├── ANN_residual_hist_train.jpg
│   │   │       │       │   └── ANN_residual_kde_train.jpg
│   │   │       │       └── valid
│   │   │       │           ├── ANN_mae_scatter.jpg
│   │   │       │           ├── ANN_mse_scatter.jpg
│   │   │       │           ├── ANN_residual_hist.jpg
│   │   │       │           └── ANN_residual_kde.jpg
│   │   │       ├── CatBoost
│   │   │       │   ├── CatBoost_feature_importance.jpg
│   │   │       │   └── full
│   │   │       │       ├── train
│   │   │       │       │   ├── CatBoost_mae_scatter_train.jpg
│   │   │       │       │   ├── CatBoost_mse_scatter_train.jpg
│   │   │       │       │   ├── CatBoost_residual_hist_train.jpg
│   │   │       │       │   └── CatBoost_residual_kde_train.jpg
│   │   │       │       └── valid
│   │   │       │           ├── CatBoost_mae_scatter.jpg
│   │   │       │           ├── CatBoost_mse_scatter.jpg
│   │   │       │           ├── CatBoost_residual_hist.jpg
│   │   │       │           └── CatBoost_residual_kde.jpg
│   │   │       ├── DT
│   │   │       │   ├── DT_feature_importance.jpg
│   │   │       │   └── full
│   │   │       │       ├── train
│   │   │       │       │   ├── DT_mae_scatter_train.jpg
│   │   │       │       │   ├── DT_mse_scatter_train.jpg
│   │   │       │       │   ├── DT_residual_hist_train.jpg
│   │   │       │       │   └── DT_residual_kde_train.jpg
│   │   │       │       └── valid
│   │   │       │           ├── DT_mae_scatter.jpg
│   │   │       │           ├── DT_mse_scatter.jpg
│   │   │       │           ├── DT_residual_hist.jpg
│   │   │       │           └── DT_residual_kde.jpg
│   │   │       └── RF
│   │   │           ├── full
│   │   │           │   ├── train
│   │   │           │   │   ├── RF_mae_scatter_train.jpg
│   │   │           │   │   ├── RF_mse_scatter_train.jpg
│   │   │           │   │   ├── RF_residual_hist_train.jpg
│   │   │           │   │   └── RF_residual_kde_train.jpg
│   │   │           │   └── valid
│   │   │           │       ├── RF_mae_scatter.jpg
│   │   │           │       ├── RF_mse_scatter.jpg
│   │   │           │       ├── RF_residual_hist.jpg
│   │   │           │       └── RF_residual_kde.jpg
│   │   │           └── RF_feature_importance.jpg
│   │   ├── Dataset_20250205_final.zip
│   │   └── fig_20250211.tar.gz
│   ├── figures.zip
│   ├── __init__.py
│   ├── metrics.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   ├── metrics.cpython-38.pyc
│   │   └── visualization.cpython-38.pyc
│   └── visualization.py
├── inference.py
├── losses
│   ├── __init__.py
│   ├── placeholder_loss.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   └── torch_losses.cpython-38.pyc
│   └── torch_losses.py
├── main.py
├── models
│   ├── ANN
│   │   ├── best_ann.pt
│   │   ├── scaler_x_ANN.pkl
│   │   ├── scaler_y_ANN.pkl
│   │   ├── x_col_names.npy
│   │   └── y_col_names.npy
│   ├── CatBoost
│   │   ├── scaler_x_CatBoost.pkl
│   │   ├── scaler_y_CatBoost.pkl
│   │   ├── trained_model.pkl
│   │   ├── x_col_names.npy
│   │   └── y_col_names.npy
│   ├── DT
│   │   ├── scaler_x_DT.pkl
│   │   ├── scaler_y_DT.pkl
│   │   ├── trained_model.pkl
│   │   ├── x_col_names.npy
│   │   └── y_col_names.npy
│   ├── __init__.py
│   ├── model_ann.py
│   ├── model_catboost.py
│   ├── model_dt.py
│   ├── model_gan.py
│   ├── model_rf.py
│   ├── model_xgb.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   ├── model_ann.cpython-38.pyc
│   │   ├── model_catboost.cpython-38.pyc
│   │   ├── model_dt.cpython-38.pyc
│   │   ├── model_gan.cpython-38.pyc
│   │   ├── model_rf.cpython-38.pyc
│   │   └── model_xgb.cpython-38.pyc
│   └── RF
│       ├── scaler_x_RF.pkl
│       ├── scaler_y_RF.pkl
│       ├── trained_model.pkl
│       ├── x_col_names.npy
│       └── y_col_names.npy
├── performance.py
├── __pycache__
│   └── main.cpython-38-pytest-7.1.2.pyc
├── README.md
├── requirements.txt
└── trainers
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-38.pyc
    │   ├── train_sklearn.cpython-38.pyc
    │   └── train_torch.cpython-38.pyc
    ├── train_sklearn.py
    └── train_torch.py

46 directories, 137 files

```
data:
#  path: "./data/Dataset_20250205_without1.csv"   # CSV file path
  path: "./data/Dataset_20250205_final.csv"   # CSV file path
#  path: "./data/Dataset_20250210_log.csv"
#  path: "./data/Dataset_20250210_Cat.csv"

  test_size: 0.2
  random_seed: 42
# 如果要用K折，把下面改成 true，并指定k_folds
  use_k_fold: false
  k_folds: 5

preprocessing:
  standardize_input: true
  standardize_output: true

model:
  # 一次性训练多个模型
  types: ["RF", "CatBoost", "DT", "ANN"]
#  types: ["RF", "CatBoost", "DT"]
#  types: ["ANN"]
  # ------------------- ANN 参数 (优化泛化能力) -------------------
  ann_params:
    input_dim: 10
    output_dim: 4
    hidden_dims: [16, 32, 32, 32, 16]        # 降低网络复杂度，减少参数数量
    dropout: 0.001                 # 增加 Dropout 防止过拟合
    learning_rate: 2e-3          # 降低学习率，减少过拟合
    epochs: 600                   # 适当增加训练轮数
    batch_size: 128              # 增大 batch_size，提高稳定性
    weight_decay: 5e-3           # 增强 L2 正则
    checkpoint_path: "./models/ANN/best_ann.pt"
    early_stopping: true         # 启用早停
    patience: 60                 # 允许更多的耐心等待
    optimizer: "AdamW"           # 使用 AdamW，改善正则化
    activation: "ReLU"           # 使用 ReLU
    random_seed: 777

  # ------------------- DT 参数 (优化深度与剪枝) -------------------
  dt_params:
    max_depth: 10              # 适当降低深度，提高泛化能力
    random_state: 42
    ccp_alpha: 0.05           # 增强剪枝力度，提高泛化能力

  # ------------------- RF 参数 (优化超参数，提升泛化) -------------------
  rf_params:
    n_estimators: 100         # 增加树的数量，提高泛化能力
    max_depth: 8             # 适当提高深度
    random_state: 42
    ccp_alpha: 0.005          # 适度降低剪枝强度
    min_samples_leaf: 2       # 适中，减少过拟合但不影响表现

  # ------------------- CatBoost 参数 (优化学习率，提高泛化) -------------------
  catboost_params:
    iterations: 500           # 增加迭代次数，提高表现
    learning_rate: 0.08       # 学习率适当回升，避免训练不足
    depth: 9                  # 维持较优深度
    random_seed: 42
    l2_leaf_reg: 9.0          # 适度降低 L2 正则，提高表达能力

  # ------------------- XGBoost 参数 (优化正则化，提升表现) -------------------
  xgb_params:
    n_estimators: 400         # 增加基学习器数量，提高泛化能力
    learning_rate: 0.03       # 降低学习率，减少过拟合
    max_depth: 6              # 适当降低深度，提高泛化能力
    random_seed: 42
    reg_alpha: 0.9            # 增加 L1 正则，防止过拟合
    reg_lambda: 1.7           # 适度降低 L2 正则，提高模型表现


loss:
  type: "mse"

training:
  log_interval: 5

evaluation:
  save_loss_curve: true
  save_scatter_mse_plot: true
  save_scatter_mae_plot: true
  save_residual_hist: true
  save_residual_kde: true
  save_heatmap: true
#  save_partial_dependence: false
  save_correlation: true
  save_feature_importance_bar: true
  save_models_evaluation_bar: true
  # ★ 新增: 是否画数据分析可视化(比如KDE,散点,盒须图等)
  save_data_analysis_plots: true

# ★ 新增: inference 配置
inference:
  # 指定推理时要用哪些模型(可能与训练时不同)
  models: [ "ANN", "CatBoost" ]   # 例如只想推理这两个
  max_combinations: 20           # 若分类组合数 > 20, 则截断简化
  # 你也可以加 output_dir, 但下面逻辑里我们自动放在 <csv_name>/inference/<model_type> 下
```



```
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
```



```
"""
data_preprocessing/data_split.py

Contains the function to split data into train/validation sets
using scikit-learn's train_test_split.
Split data into train/val sets or K folds
"""

from sklearn.model_selection import train_test_split, KFold

def split_data(X, Y, test_size=0.2, random_state=42):
    """
    Split the dataset into train and validation sets.
    :param X: Input features, shape (N, input_dim)
    :param Y: Output targets, shape (N, output_dim)
    :param test_size: Fraction of data for validation
    :param random_state: For reproducibility
    :return: X_train, X_val, Y_train, Y_val
    """
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)


def kfold_split_data(X, Y, n_splits=5, random_state=42, shuffle=True):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        yield (X_train, X_val, Y_train, Y_val)
```

```
"""
data_preprocessing/my_dataset.py

Defines a custom PyTorch Dataset to facilitate DataLoader usage
when training with PyTorch-based models (like the ANN).
"""

import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
    A simple Dataset for multi-output regression using PyTorch.
    """
    def __init__(self, X, Y):
        """
        Constructor for MyDataset.
        :param X: NumPy array of input features
        :param Y: NumPy array of output targets
        """
        # Convert to torch.float32 Tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Return a single sample (x, y) at index idx.
        """
        return self.X[idx], self.Y[idx]
```

```
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
        scaler_y = StandardScaler()
        scaler_y.fit(Y_train_s)
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
```



```
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
```



```
"""
visualization.py

Contains functions for plotting:
- Loss curve (train/val)
- Scatter plot (True vs. Pred) with the colorbar of mse mae etc
- Residual histograms
- Residual KDE
- Mixed correlation heatmap
- 2D heatmap scanning 2 input features
All figures are saved as JPG (DPI=700) to ./evaluation/figures/.
"""

import os
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.collections import PolyCollection
import pandas as pd
import math
import scipy.stats as ss
from matplotlib.patches import Patch
from sklearn.metrics import r2_score

FIG_DIR = "./evaluation/figures"

# 1) 全局设置字体大小 & 加粗
matplotlib.rcParams['font.size'] = 13  # 基础字号
matplotlib.rcParams['axes.labelsize'] = 13
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titlesize'] = 15
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 12
# 你也可以加更多 rcParams，比如 'lines.linewidth'=1.5 等

def cramers_v(x, y):
    """
    Cramér's V for categorical-categorical correlation.
    x, y are 1D arrays (list-like) of categorical variables
    Returns: float in [0, 1]
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    # Bias correction
    phi2 = max(0, chi2 - (k - 1) * (r - 1) / (n - 1))
    r_adj = r - (r - 1) ** 2 / (n - 1)
    k_adj = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_adj - 1, r_adj - 1)
    if denom == 0:
        return 0.0
    else:
        return math.sqrt(phi2 / denom)


def correlation_ratio(categories, values):
    """
    Correlation Ratio (η) for categorical -> numeric relationship.
    categories: 1D array of categorical data
    values: 1D array of numeric data
    Returns: float in [0, 1]

    如果 categories 中有类别没有样本，或者 values 全为 NaN，会出问题。
    你可以根据实际情况做更多容错处理。
    """
    # 将数据组合成 dataframe 以方便处理
    df = pd.DataFrame({'cat': categories, 'val': values})
    # 去掉 NaN
    df.dropna(inplace=True)

    # 各类别组的均值
    group_means = df.groupby('cat')['val'].mean()
    # 总均值
    mean_all = df['val'].mean()

    # 分子
    ss_between = 0
    # 分母
    ss_total = 0

    for cat_value, group_mean in group_means.items():
        group = df[df['cat'] == cat_value]
        n = len(group)
        ss_between += n * (group_mean - mean_all) ** 2

    ss_total = ((df['val'] - mean_all) ** 2).sum()

    if ss_total == 0:
        return 0.0

    eta = math.sqrt(ss_between / ss_total)
    return eta


def mixed_correlation_matrix(X, col_names, numeric_cols_idx, method_numeric="pearson", dropna=True):
    """
    计算混合变量相关性矩阵（数值-数值, 类别-类别, 数值-类别）。
    X: numpy.ndarray 或者 pandas.DataFrame, shape = (n_samples, n_features)
    col_names: 每列的名字(list)
    numeric_cols_idx: 哪些列是数值列 (list of int)
    method_numeric: 对数值-数值的相关系数, 可选"pearson", "spearman", ...
    dropna: 是否在计算相关时丢弃含 NaN 的样本

    返回 (corr_matrix, used_method_matrix):
      corr_matrix: shape = (n_features, n_features)
      used_method_matrix: 每个 [i, j] 存放使用了什么方法 (str)
    """
    n_features = X.shape[1]

    # 如果 X 是 np.ndarray, 先转换成 DataFrame 方便按列操作
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=col_names)

    # 将 numeric_cols_idx 转为 set，方便判断
    numeric_cols_set = set(numeric_cols_idx)

    # 最终的相关性矩阵 & 方法矩阵
    corr_matrix = np.zeros((n_features, n_features), dtype=float)
    used_method_matrix = [["" for _ in range(n_features)] for _ in range(n_features)]

    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                corr_matrix[i, j] = 1.0
                used_method_matrix[i][j] = "self"
                continue
            if i > j:
                # 确保对称性
                corr_matrix[i, j] = corr_matrix[j, i]
                used_method_matrix[i][j] = used_method_matrix[j][i]
                continue

            col_i = col_names[i]
            col_j = col_names[j]

            # 取出列数据
            data_i = X[col_i]
            data_j = X[col_j]

            # 如果需要，drop NaN
            if dropna:
                valid_mask = ~data_i.isna() & ~data_j.isna()
                data_i = data_i[valid_mask]
                data_j = data_j[valid_mask]

            # 判断列类型: numeric or categorical
            i_is_num = (i in numeric_cols_set)
            j_is_num = (j in numeric_cols_set)

            if i_is_num and j_is_num:
                # 数值-数值 => 采用 method_numeric (默认 pearson)
                if method_numeric.lower() == "pearson":
                    r, _ = ss.pearsonr(data_i, data_j)
                    used_method = "pearson"
                elif method_numeric.lower() == "spearman":
                    r, _ = ss.spearmanr(data_i, data_j)
                    used_method = "spearman"
                elif method_numeric.lower() == "kendall":
                    r, _ = ss.kendalltau(data_i, data_j)
                    used_method = "kendall"
                else:
                    raise ValueError(f"Unsupported numeric correlation method: {method_numeric}")
                corr_matrix[i, j] = r
                used_method_matrix[i][j] = used_method

            elif (not i_is_num) and (not j_is_num):
                # 类别-类别 => Cramér's V
                r = cramers_v(data_i, data_j)
                corr_matrix[i, j] = r
                used_method_matrix[i][j] = "cramers_v"
            else:
                # 一个数值, 一个分类 => correlation ratio
                # 约定: 如果 i 是分类, j 是数值 => cat->numeric
                # 或反过来都一样, correlation ratio 也可看做对称使用
                if i_is_num:  # swap
                    cat_data, num_data = data_j, data_i
                else:
                    cat_data, num_data = data_i, data_j
                r = correlation_ratio(cat_data, num_data)
                corr_matrix[i, j] = r
                used_method_matrix[i][j] = "corr_ratio"

    # 因为我们只计算了上三角，所以再对称一下
    for i in range(n_features):
        for j in range(i):
            corr_matrix[i, j] = corr_matrix[j, i]
            used_method_matrix[i][j] = used_method_matrix[j][i]

    return corr_matrix, used_method_matrix


def ensure_fig_dir(dir_path=FIG_DIR):
    """
    Ensure that the directory 'dir_path' exists.
    If not provided, default is FIG_DIR = './evaluation/figures'.
    """
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def plot_loss_curve(train_losses, val_losses, filename="loss_curve.jpg"):
    """
    Plot the training and validation loss vs. epochs.
    """
    ensure_fig_dir()
    plt.figure()
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training/Validation Loss")
    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=700, format='jpg')
    plt.close()

def plot_scatter_3d_outputs_mse(y_true, y_pred, y_labels=None, filename="scatter_3d_output_mse.jpg"):
    """
    Create subplots for multi-output regression (dimension = n_outputs).
    Each subplot:
      - X轴: True Y
      - Y轴: Predicted Y
      - Color: MSE (squared error) per sample for that output dimension.
      - 额外: 在标题处显示该维度的 R² 值.
    """
    ensure_fig_dir()

    if y_pred.ndim != 2:
        raise ValueError(f"y_pred shape expected (N, n_outputs), got {y_pred.shape}")
    n_samples, n_outputs = y_pred.shape

    fig, axes = plt.subplots(1, n_outputs, figsize=(4 * n_outputs, 4), squeeze=False)

    for i in range(n_outputs):
        # ========== 计算每个维度的MSE误差(每个点) ==========
        errors = (y_true[:, i] - y_pred[:, i]) ** 2

        # ========== 计算该维度的R² ==========
        r2_val = r2_score(y_true[:, i], y_pred[:, i])

        ax = axes[0, i]
        sc = ax.scatter(
            y_true[:, i],
            y_pred[:, i],
            c=errors,
            alpha=0.5,
            cmap='brg'
        )
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)

        # ========== 在子图标题处加 R² ==========
        if y_labels and i < len(y_labels):
            title_str = f"{y_labels[i]} (MSE colormap)\nR² = {r2_val:.3f}"
            ax.set_xlabel(f"True {y_labels[i]}")
            ax.set_ylabel(f"Pred {y_labels[i]}")
        else:
            title_str = f"Output {i} (MSE colormap)\nR² = {r2_val:.3f}"
            ax.set_xlabel("True")
            ax.set_ylabel("Pred")

        ax.set_title(title_str)

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Squared Error", fontweight='bold', fontsize=13)
        cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=700, format='jpg')
    plt.close()

def plot_scatter_3d_outputs_mae(y_true, y_pred, y_labels=None, filename="scatter_3d_output_mae.jpg"):
    """
    Similar to plot_scatter_3d_outputs_mse, but color = absolute error.
    Additionally, show R² in the subplot title.
    """
    ensure_fig_dir()

    if y_pred.ndim != 2:
        raise ValueError(f"y_pred shape expected (N, n_outputs), got {y_pred.shape}")
    n_samples, n_outputs = y_pred.shape

    fig, axes = plt.subplots(1, n_outputs, figsize=(4 * n_outputs, 4), squeeze=False)

    for i in range(n_outputs):
        # ========== 计算该维度的MAE误差(每个点) ==========
        errors = np.abs(y_true[:, i] - y_pred[:, i])

        # ========== 计算该维度的R² ==========
        r2_val = r2_score(y_true[:, i], y_pred[:, i])

        ax = axes[0, i]
        sc = ax.scatter(
            y_true[:, i],
            y_pred[:, i],
            c=errors,
            alpha=0.5,
            cmap='ocean'
        )
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)

        # ========== 在子图标题处加 R² ==========
        if y_labels and i < len(y_labels):
            title_str = f"{y_labels[i]} (MAE colormap)\nR² = {r2_val:.3f}"
            ax.set_xlabel(f"True {y_labels[i]}")
            ax.set_ylabel(f"Pred {y_labels[i]}")
        else:
            title_str = f"Output {i} (MAE colormap)\nR² = {r2_val:.3f}"
            ax.set_xlabel("True")
            ax.set_ylabel("Pred")

        ax.set_title(title_str)

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Absolute Error", fontweight='bold', fontsize=13)
        cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=700, format='jpg')
    plt.close()

def plot_residual_histogram(
        y_true, y_pred, y_labels=None,
        cmap_name="coolwarm",
        vmin=-45, vmax=45,
        filename="residual_hist_bottom.jpg"
):
    """
    Plot histograms of residuals for each output dimension,
    with a shared horizontal colorbar at the bottom (with some padding/spacing).
    Font is bigger and bold.
    """
    ensure_fig_dir()

    residuals = y_true - y_pred
    n_outputs = residuals.shape[1]

    fig, axes = plt.subplots(1, n_outputs, figsize=(4 * n_outputs, 4.5))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.88, wspace=0.3)

    # 准备统一的bin边界 => 30个bin,范围[-45, 45]
    num_bins = 30
    bins_array = np.linspace(vmin, vmax, num_bins + 1)

    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i in range(n_outputs):
        ax = axes[i] if n_outputs > 1 else axes

        # 使用 bins=bins_array 确保每张图bin一致
        hist_data, bin_edges, patches = ax.hist(
            residuals[:, i],
            bins=bins_array,
            alpha=0.9,
            edgecolor='none'
        )

        # 给每个bin赋颜色
        for b_idx, patch in enumerate(patches):
            bin_center = 0.5 * (bin_edges[b_idx] + bin_edges[b_idx + 1])
            patch.set_facecolor(cmap(norm(bin_center)))

        # 标题或标签
        if y_labels and i < len(y_labels):
            ax.set_title(f"Residuals of {y_labels[i]}")
        else:
            ax.set_title(f"Output {i} Residual")

        ax.set_xlabel("Residual")
        ax.set_ylabel("Count")
        ax.set_xlim(vmin, vmax)

    # colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.07,
        pad=0.20,
        shrink=0.9
    )
    cbar.set_label("Residual Value", fontweight='bold', fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[plot_residual_histogram] => {save_path}")


def plot_residual_kde(
    y_true, y_pred, y_labels=None,
    cmap_name="coolwarm",
    vmin=-45, vmax=45,
    filename="residual_kde_bottom.jpg"
):
    """
    Plot KDE of residuals for each output dimension,
    with color fill based on x (residual).
    Shared horizontal colorbar at bottom,
    bigger bold fonts, same layout as residual_histogram.
    """
    ensure_fig_dir()

    residuals = y_true - y_pred
    n_outputs = residuals.shape[1]

    fig, axes = plt.subplots(1, n_outputs, figsize=(4*n_outputs, 4.5))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.88, wspace=0.3)

    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i in range(n_outputs):
        ax = axes[i] if n_outputs>1 else axes

        sns.kdeplot(residuals[:, i], ax=ax, fill=False, color="black")
        lines = ax.get_lines()
        if not lines:
            continue
        line = lines[-1]
        x_plot = line.get_xdata()
        y_plot = line.get_ydata()

        idxsort = np.argsort(x_plot)
        x_plot = x_plot[idxsort]
        y_plot = y_plot[idxsort]

        # 分段填充
        for j in range(len(x_plot)-1):
            x0, x1 = x_plot[j], x_plot[j+1]
            y0, y1 = y_plot[j], y_plot[j+1]
            xmid = 0.5*(x0 + x1)
            color = cmap(norm(xmid))

            verts = np.array([
                [x0, 0],
                [x0, y0],
                [x1, y1],
                [x1, 0]
            ])
            poly = PolyCollection([verts], facecolors=[color], edgecolor='none', alpha=0.6)
            ax.add_collection(poly)

        if y_labels and i < len(y_labels):
            ax.set_title(f"Residual KDE of {y_labels[i]}")
        else:
            ax.set_title(f"KDE - Output {i}")

        ax.set_xlabel("Residual")
        ax.set_ylabel("Density")
        ax.set_xlim(vmin, vmax)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.07,
        pad=0.20,
        shrink=0.9
    )
    cbar.set_label("Residual Value", fontweight='bold', fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[plot_residual_kde] => {save_path}")

def plot_2d_heatmap(model,
                    x_range=(0, 1),
                    y_range=(0, 1),
                    resolution=50,
                    # 新增参数
                    num_features=100,  # X_train_s.shape[1], 真实输入总维
                    col_x=0,  # one-hot后想扫描的第1个 numeric col index
                    col_y=1,  # one-hot后想扫描的第2个 numeric col index
                    fixed_input=None,  # 若不提供则默认全0
                    input_scaler=None,
                    output_scaler=None,
                    output_index=0,
                    filename="heatmap.jpg"):
    """
    Scan 2 specific numeric columns in [x_range, y_range],
    fix the other (num_features-2) dims to default 0 or user-provided 'fixed_input'.
    """
    ensure_fig_dir()

    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    Z = np.zeros((resolution, resolution))

    if fixed_input is None:
        base_vec = np.zeros(num_features, dtype=float)
    else:
        assert len(fixed_input) == num_features, "fixed_input must match num_features"
        base_vec = fixed_input.copy()

    for i, xv in enumerate(x_vals):
        for j, yv in enumerate(y_vals):
            inp_vec = base_vec.copy()
            inp_vec[col_x] = xv
            inp_vec[col_y] = yv

            inp_2d = inp_vec[np.newaxis, :]

            if input_scaler is not None:
                inp_2d = input_scaler.transform(inp_2d)

            pred = model.predict(inp_2d)
            if output_scaler is not None:
                pred = output_scaler.inverse_transform(pred)

            Z[j, i] = pred[0, output_index]

    plt.figure(figsize=(6, 5))
    sns.heatmap(Z, xticklabels=False, yticklabels=False, cmap="viridis")
    plt.title(f"Heatmap - Output {output_index}", fontweight='bold')
    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=700, format='jpg')
    plt.close()
    print(f"Saved heatmap to {save_path}")


def plot_correlation_heatmap(
    X,
    col_names,
    numeric_cols_idx,
    filename="correlation_heatmap.jpg",
    method_numeric="pearson",
    cmap="ocean",
    vmin=-1,
    vmax=1
):
    """
    绘制混合变量相关性热力图，并保存。
    """

    ensure_fig_dir()  # 确保输出目录存在

    corr_matrix, used_methods = mixed_correlation_matrix(
        X, col_names, numeric_cols_idx,
        method_numeric=method_numeric
    )

    # 1) 设置一个相对较大的画布
    #    这里使用 max(10, 0.5*len(col_names)) 只是示例，你可适度调整
    fig, ax = plt.subplots(
        figsize=(max(10, 0.5 * len(col_names)), max(8, 0.5 * len(col_names)))
    )

    # 2) 绘制热力图
    #    注意：此处传 ax=ax，从而可在 ax 上进行更多控制
    sns.heatmap(
        corr_matrix,
        xticklabels=col_names,
        yticklabels=col_names,
        cmap=cmap,
        annot=False,
        square=True,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={"shrink": 0.8, "aspect": 30, "label": "Correlation"}
    )

    # 3) 设置标题，x轴、y轴刻度字体
    ax.set_title("Mixed Correlation Heatmap", fontsize=14)
    # 让 x 轴标签倾斜，避免重叠
    # ha='right' 配合 rotation=45 可以保证长标签更容易看清
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    plt.setp(ax.get_yticklabels(), rotation=0,  ha="right", fontsize=12)

    # 4) 做一次紧凑布局，防止标签被裁剪
    plt.tight_layout()

    # 5) 保存
    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=700)
    plt.close()
    print(f"[plot_correlation_heatmap] Saved => {save_path}")



# ---------------------
# 新增: RF/用于随机森林/决策树/GBM等的特征重要度柱状图
# ---------------------
def plot_rf_feature_importance_bar(
        rf_model,
        feature_names,
        filename="rf_feature_importance_bar.jpg",
        top_k=20,
        threshold=0.05
):
    """
    使用柱状图（bar chart）绘制随机森林特征重要度，仅显示最重要的 top_k 个特征。
    - 重要度 **> threshold** 的特征 **红色**
    - 重要度 **≤ threshold** 的特征 **蓝色**
    - **浅灰色区域** 表示 `Feature Importance < threshold`
    - **添加图例**

    Args:
        rf_model (RandomForestRegressor): 训练好的随机森林模型
        feature_names (List[str]): 特征名称列表（与训练数据列对应）
        filename (str): 输出图像文件名
        top_k (int): 只显示前 top_k 个特征
        threshold (float): 特征重要度的阈值，超过该值的标红色，低于该值的标蓝色
    """
    # 确保保存目录存在
    ensure_fig_dir()

    # 确保有特征重要度数据
    importances = rf_model.feature_importances_
    #============testing===============
    # print("Feature importances (should sum to 1):", importances.sum())
    # print("Min importance:", importances.min(), "Max importance:", importances.max())
    #
    # if not np.isclose(importances.sum(), 1.0):
    #     print("⚠ Warning: Feature importances are not normalized correctly!")
    #===========testing end============

    if importances is None or len(importances) == 0:
        print("[WARNING] No importances found or importances is empty!")
        return

    # 1. 对特征重要度排序（从大到小）
    sorted_indices = np.argsort(importances)[::-1]
    # 2. 选取前 top_k 个特征
    topk_indices = sorted_indices[:top_k]

    # 取出对应的特征名 & 特征重要度
    topk_features = [feature_names[i] for i in topk_indices]
    topk_importances = importances[topk_indices]

    # 3. 颜色设置：大于 threshold 为红色，小于等于 threshold 为蓝色
    colors = ["red" if imp > threshold else "blue" for imp in topk_importances]

    # 4. 创建画布
    fig, ax = plt.subplots(figsize=(8, 6))

    # 5. 画柱状图
    bars = ax.barh(range(len(topk_importances)), topk_importances, align="center", color=colors)

    # 6. 设置 y 轴为特征名
    ax.set_yticks(range(len(topk_importances)))
    ax.set_yticklabels(topk_features, fontsize=10)

    # 7. 让重要度最高的特征排在最上面
    ax.invert_yaxis()

    # 8. 设置标题与轴标签
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Feature Importance (Top-{top_k})", fontsize=14, fontweight='bold')

    # 9. 添加浅灰色区域 (highlight region where Feature Importance < threshold)
    ax.axvspan(0, threshold, facecolor='lightgray', alpha=0.5)

    # 10. 添加红色虚线标记 threshold
    ax.axvline(x=threshold, color='red', linestyle='dashed', linewidth=2)

    # 11. 添加图例
    legend_elements = [
        Patch(facecolor="red", label=f"Importance > {threshold}"),
        Patch(facecolor="royalblue", label=f"Importance ≤ {threshold}")
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=12)

    # 12. 布局优化 & 保存
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=700)
    plt.close()

    print(f"[plot_rf_feature_importance_bar] Saved => {save_path}")

# ============ 新增: 三指标横向对比图 ============

def plot_hbar_with_mean(
    ax,
    model_names,
    values,
    subplot_label="(a)",
    metric_label="Metric",
    bigger_is_better=False,
    width=0.4
):
    arr = np.array(values)
    if bigger_is_better:
        best_idx = arr.argmax()
        worst_idx = arr.argmin()
    else:
        best_idx = arr.argmin()
        worst_idx = arr.argmax()

    colors = []
    for i, val in enumerate(arr):
        if i == best_idx:
            colors.append("red")
        elif i == worst_idx:
            colors.append("blue")
        else:
            colors.append("green")

    mean_val = arr.mean()
    y_positions = np.arange(len(arr))
    bars = ax.barh(y_positions, arr, color=colors, alpha=0.8, height=width)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_names)
    ax.invert_yaxis()
    ax.text(-0.08, 1.05, subplot_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=14, fontweight="bold")
    ax.set_title(metric_label, fontsize=14, fontweight='bold')

    for i, v in enumerate(arr):
        ax.text(v, i, f"{v:.2f}", ha=("left" if v>=0 else "right"),
                va="center", fontsize=10)

    ax.axvline(mean_val, color='gray', linestyle='--', linewidth=2)
    xmin, xmax = sorted([0, mean_val])
    ax.axvspan(xmin, xmax, facecolor='gray', alpha=0.2)

    max_val = arr.max()
    min_val = arr.min()
    if min_val < 0:
        ax.set_xlim(min_val * 1.1, max_val * 1.79)
    else:
        ax.set_xlim(0, max_val * 1.79)

    legend_elements = [
        Patch(facecolor="red", label="Best"),
        Patch(facecolor="blue", label="Worst"),
        Patch(facecolor="green", label="Ordinary"),
        Patch(facecolor="gray", alpha=0.2, label="Under Mean"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")


def plot_hbar_with_threshold(
        ax,
        model_names,
        values,
        subplot_label="(a)",
        metric_label="Metric",
        bigger_is_better=False,
        width=0.4,
        threshold_h=0.5,
        threshold_l=0.0
):
    arr = np.array(values)
    if bigger_is_better:
        best_idx = arr.argmax()
        worst_idx = arr.argmin()
    else:
        best_idx = arr.argmin()
        worst_idx = arr.argmax()

    colors = []
    for i, val in enumerate(arr):
        if i == best_idx:
            colors.append("red")
        elif i == worst_idx:
            colors.append("blue")
        else:
            colors.append("green")

    y_positions = np.arange(len(arr))
    bars = ax.barh(y_positions, arr, color=colors, alpha=0.8, height=width)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_names)
    ax.invert_yaxis()
    ax.text(-0.08, 1.05, subplot_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=14, fontweight="bold")
    ax.set_title(metric_label, fontsize=14, fontweight='bold')

    for i, v in enumerate(arr):
        ax.text(v, i, f"{v:.2f}", ha=("left" if v >= 0 else "right"),
                va="center", fontsize=10)

    if threshold_l == 0.0:
        ax.axvspan(threshold_l, threshold_h, facecolor='gray', alpha=0.2)
        legend_elements = [
            Patch(facecolor="gray", alpha=0.2, label="Acceptable")
        ]
    else:
        ax.axvspan(0, threshold_l, facecolor='gray', alpha=0.2)
        ax.axvspan(threshold_l, threshold_h, facecolor='lightcoral', alpha=0.3)
        ax.axvline(threshold_l, color='gray', linestyle='--', linewidth=2)
        ax.axvline(threshold_h, color='gray', linestyle='--', linewidth=2)
        legend_elements = [
            Patch(facecolor="gray", alpha=0.2, label="Acceptable"),
            Patch(facecolor="lightcoral", alpha=0.3, label="Overfitting Risk")
        ]

    max_val = arr.max()
    min_val = arr.min()
    if min_val < 0:
        ax.set_xlim(min_val * 1.1, max_val * 1.79)
    else:
        ax.set_xlim(0, max_val * 1.79)

    legend_elements.extend([
        Patch(facecolor="red", label="Best"),
        Patch(facecolor="blue", label="Worst"),
        Patch(facecolor="green", label="Ordinary")
    ])
    ax.legend(handles=legend_elements, loc="lower right")


def plot_three_metrics_horizontal(
    metrics_data,
    save_name="three_metrics_horizontal.jpg"
):
    ensure_fig_dir()
    model_names = list(metrics_data.keys())
    mse_vals = [metrics_data[m]["MSE"] for m in model_names]
    mae_vals = [metrics_data[m]["MAE"] for m in model_names]
    r2_vals  = [metrics_data[m]["R2"]  for m in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) MSE => lower better
    plot_hbar_with_mean(
        ax=axes[0],
        model_names=model_names,
        values=mse_vals,
        subplot_label="(a)",
        metric_label="MSE (Lower is better)",
        bigger_is_better=False,
        width=0.4
    )
    # (b) MAE => lower better
    plot_hbar_with_mean(
        ax=axes[1],
        model_names=model_names,
        values=mae_vals,
        subplot_label="(b)",
        metric_label="MAE (Lower is better)",
        bigger_is_better=False,
        width=0.4
    )
    # (c) R2 => higher better
    plot_hbar_with_mean(
        ax=axes[2],
        model_names=model_names,
        values=r2_vals,
        subplot_label="(c)",
        metric_label="R2 (Higher is better)",
        bigger_is_better=True,
        width=0.4
    )
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot_three_metrics_horizontal] => {out_path}")


def plot_overfitting_horizontal(
    overfit_data,
    save_name="overfitting_horizontal.jpg"
):
    """
    将所有模型的过拟合指标(MSE_ratio 与 R2_diff)一起绘制成两列横向条形图。
    - MSE_ratio = ValMSE / TrainMSE (越小越好, 1 => 无差别, >1 => 过拟合)
    - R2_diff   = TrainR2 - ValR2   (越小越好, 0 => 无差别, >0 => 过拟合)

    :param overfit_data: dict, 形如:
        {
          "ModelA": {"MSE_ratio": 1.23, "R2_diff": 0.10},
          "ModelB": {"MSE_ratio": 0.95, "R2_diff": -0.02},
          ...
        }
    :param save_name: 输出文件名
    """
    ensure_fig_dir()
    model_names = list(overfit_data.keys())

    # 从 overfit_data 取出 MSE_ratio 和 R2_diff
    msr_vals = [overfit_data[m]["MSE_ratio"] for m in model_names]
    r2d_vals = [overfit_data[m]["R2_diff"]   for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) MSE_ratio => bigger =>更多过拟合 => "Lower is better"
    plot_hbar_with_threshold(
        ax=axes[0],
        model_names=model_names,
        values=msr_vals,
        subplot_label="(a)",
        metric_label="MSE Ratio (Val/Train)\n(Lower is better)",
        bigger_is_better=False,  # 我们希望越小越好
        width=0.4,
        threshold_h=10
    )

    # (b) R2_diff => (TrainR2 - ValR2) => bigger =>更多过拟合 => "Lower is better"
    plot_hbar_with_threshold(
        ax=axes[1],
        model_names=model_names,
        values=r2d_vals,
        subplot_label="(b)",
        metric_label="R2 diff (Train - Val)\n(Lower is better)",
        bigger_is_better=False,  # 同理：越小越好
        width=0.4,
        threshold_h=0.2,
        threshold_l=0.15
    )

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot_overfitting_horizontal] => {out_path}")


def plot_2d_mimo_heatmaps(
    grid_x, grid_y, predictions,
    out_labels=None,
    out_dir="./",
    prefix="mimo_heatmap"
):
    """
    多输出回归结果的 2D 网格热力图
    """
    H, W, out_dim = predictions.shape
    if out_labels is None or len(out_labels)!=out_dim:
        out_labels = [f"Output_{i+1}" for i in range(out_dim)]
    for i in range(out_dim):
        fig, ax = plt.subplots(figsize=(6,5))
        z = predictions[:,:,i]
        cm_ = ax.pcolormesh(grid_x, grid_y, z, shading='auto', cmap='viridis')
        fig.colorbar(cm_, ax=ax, label=f"{out_labels[i]}")
        ax.set_xlabel("potential")
        ax.set_ylabel("catalyst size")
        ax.set_title(f"Heatmap of {out_labels[i]}")
        out_fn = os.path.join(out_dir, f"{prefix}_{i+1}.jpg")
        plt.savefig(out_fn, dpi=150, bbox_inches='tight')
        plt.close()


def plot_kde_distribution(df, columns, filename="kde_distribution.jpg", out_dir="./evaluation/figures"):
    """
    绘制各变量的 KDE 分布，观察偏态、双峰等。
    使用颜色填充，以更直观地展示分布曲线。
    """
    fig_dir = ensure_fig_dir(out_dir)
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 5))
    if len(columns)==1:
        axes = [axes]  # 保证可迭代

    cmap = cm.get_cmap("coolwarm")
    # 整体的最小值/最大值
    vmin = df[columns].min().min()
    vmax = df[columns].max().max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i, col in enumerate(columns):
        ax = axes[i]
        sns.kdeplot(df[col], ax=ax, fill=False, color="black")
        lines = ax.get_lines()
        if lines:
            line = lines[-1]
            x_plot, y_plot = line.get_xdata(), line.get_ydata()
            idxsort = np.argsort(x_plot)
            x_plot, y_plot = x_plot[idxsort], y_plot[idxsort]
            for j in range(len(x_plot)-1):
                x0, x1 = x_plot[j], x_plot[j+1]
                y0, y1 = y_plot[j], y_plot[j+1]
                color = cmap(norm(0.5*(x0 + x1)))  # 取中点
                verts = np.array([[x0, 0], [x0, y0], [x1, y1], [x1, 0]])
                poly = PolyCollection([verts], facecolors=[color], edgecolor='none', alpha=0.6)
                ax.add_collection(poly)
        ax.set_title(f"KDE of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
    plt.tight_layout()
    save_path = os.path.join(fig_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[plot_kde_distribution] => {save_path}")


def plot_catalyst_size_vs_product(df, filename="catalyst_size_vs_product.jpg", out_dir="./evaluation/figures"):
    """
    绘制催化剂尺寸 vs 产物产量（散点图）
    可根据 df 中的列 'Size','Catalyst','H2','CO','C1','C2+' 做区分
    """
    fig_dir = ensure_fig_dir(out_dir)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    products = ['H2', 'CO', 'C1', 'C2+']

    for i, product in enumerate(products):
        ax = axes[i // 2, i % 2]
        if all(col in df.columns for col in ['Size', 'Catalyst', product]):
            sns.scatterplot(x='Size', y=product, hue='Catalyst', data=df, ax=ax, alpha=0.7)
            ax.set_title(f'Catalyst Size vs {product} Yield')
            ax.set_xlabel('Catalyst Size (nm)')
            ax.set_ylabel(f'{product} Yield (%)')
        else:
            ax.text(0.5,0.5,f"Cols not found for {product}", ha='center', va='center')
    plt.tight_layout()
    save_path = os.path.join(fig_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[plot_catalyst_size_vs_product] => {save_path}")


def plot_potential_vs_product_by_electrolyte(df, filename="potential_vs_product_by_electrolyte.jpg",
                                             out_dir="./evaluation/figures"):
    """
    绘制不同电解质下的电位 vs 产物选择性（散点图）
    需求使用列：Potential, Anion sepcies of electrolyte, 'H2','CO','C1','C2+' ...
    """
    fig_dir = ensure_fig_dir(out_dir)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    products = ['H2', 'CO', 'C1', 'C2+']

    for i, product in enumerate(products):
        ax = axes[i // 2, i % 2]
        if all(col in df.columns for col in ['Potential', 'Anion sepcies of electrolyte', product]):
            sns.scatterplot(x='Potential', y=product, hue='Anion sepcies of electrolyte',
                            data=df, ax=ax, alpha=0.7)
            ax.set_title(f'Potential vs {product} Selectivity Across Electrolytes')
            ax.set_xlabel('Potential (V)')
            ax.set_ylabel(f'{product} Yield (%)')
        else:
            ax.text(0.5,0.5,f"Cols not found for {product}", ha='center', va='center')
    plt.tight_layout()
    save_path = os.path.join(fig_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[plot_potential_vs_product_by_electrolyte] => {save_path}")


def plot_product_distribution_by_catalyst_and_potential(df, filename="product_distribution.jpg",
                                                       out_dir="./evaluation/figures"):
    """
    绘制不同催化剂和电位下产物分布的盒须图
    需求列: Catalyst, Potential, 'H2','CO','C1','C2+'
    """
    fig_dir = ensure_fig_dir(out_dir)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    products = ['H2', 'CO', 'C1', 'C2+']

    if 'Potential' in df.columns:
        # 分 bin
        df['_Pot_bin'] = pd.cut(df['Potential'], bins=5)
    else:
        df['_Pot_bin'] = "Unknown"

    for i, product in enumerate(products):
        ax = axes[i]
        if all(col in df.columns for col in ['Catalyst', product]):
            sns.boxplot(x='Catalyst', y=product, hue='_Pot_bin', data=df, ax=ax)
            ax.set_title(f'{product} Yield by Catalyst and Potential')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5,0.5,f"Cols not found for {product}", ha='center', va='center')
    plt.tight_layout()
    save_path = os.path.join(fig_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[plot_product_distribution_by_catalyst_and_potential] => {save_path}")

# ============【汇总】一个函数 => 在 main.py 中一次性调用 =================
def plot_advanced_data_analysis(df, out_dir):
    """
    依次调用上面几个函数:
    1) KDE 分布图
    2) catalyst_size_vs_product
    3) potential_vs_product_by_electrolyte
    4) product_distribution_by_catalyst_and_potential
    """
    # 你可根据 df 实际存在的列决定要画什么
    # 例如先尝试确定 'H2','CO','C1','C2+','Size','Catalyst','Potential' 是否存在

    # 1. KDE
    # 如果 df 中有一些数值列，如 ["Potential","H2","CO","C1","C2+","Size"]，可全部放一起:
    possible_cols = [c for c in ["Potential","H2","CO","C1","C2+","Size"] if c in df.columns]
    if len(possible_cols)>0:
        plot_kde_distribution(df, possible_cols, filename="kde_distribution.jpg", out_dir=out_dir)

    # 2. Catalyst size vs product
    plot_catalyst_size_vs_product(df, filename="catalyst_size_vs_product.jpg", out_dir=out_dir)

    # 3. Potential vs product by electrolyte
    plot_potential_vs_product_by_electrolyte(df, filename="potential_vs_product_by_electrolyte.jpg", out_dir=out_dir)

    # 4. Product distribution by catalyst & potential
    plot_product_distribution_by_catalyst_and_potential(df, filename="product_distribution.jpg", out_dir=out_dir)
```



```
"""
torch_losses.py

Implements PyTorch-specific loss functions (MSE, MAE, etc.).
"""

import torch.nn.functional as F

def mse_loss(pred, target):
    """
    Mean Squared Error.
    """
    return F.mse_loss(pred, target)

def mae_loss(pred, target):
    """
    Mean Absolute Error.
    """
    return F.l1_loss(pred, target)

def get_torch_loss_fn(name="mse"):
    """
    Return the specified PyTorch loss function by name.
    :param name: "mse" or "mae"
    :return: callable that takes (pred, target) -> scalar loss
    """
    if name.lower() == "mse":
        return mse_loss
    elif name.lower() == "mae":
        return mae_loss
    else:
        raise ValueError(f"Unknown loss: {name}")
```



```
"""
models/model_ann.py

Implementation of an ANN for multi-output regression with optional Dropout.
"""
import torch
import torch.nn as nn

class ANNRegression(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=[32,64,32],
                 dropout=0.0,
                 activation="ReLU",
                 random_seed=None):
        """
        :param input_dim:  输入特征维度
        :param output_dim: 输出维度
        :param hidden_dims: 隐藏层大小列表
        :param dropout: Dropout 概率, e.g. 0.2
        :param activation: 激活函数名称, e.g. "ReLU" / "Tanh" / "Sigmoid"
        :param random_seed: 若指定, 则使用该种子来初始化网络权重
        """
        super().__init__()

        # 如果用户传了 random_seed，就在这里手动设置一下
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # 选择激活函数
        act_fn = None
        if activation.lower() == "relu":
            act_fn = nn.ReLU
        elif activation.lower() == "tanh":
            act_fn = nn.Tanh
        elif activation.lower() == "sigmoid":
            act_fn = nn.Sigmoid
        else:
            # 默认用ReLU
            act_fn = nn.ReLU

        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```



```
"""
models/model_catboost.py
"""

from catboost import CatBoostRegressor

class CatBoostRegression:
    """
    CatBoost with l2_leaf_reg controlling L2 regularization strength.
    """
    def __init__(self, iterations=100, learning_rate=0.1, depth=6,
                 random_seed=42, l2_leaf_reg=3.0):
        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_seed,
            verbose=0,
            loss_function="MultiRMSE",
            l2_leaf_reg=l2_leaf_reg
        )

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    # @property
    # def feature_importances_(self):
    #     return self.model.get_feature_importance(type="PredictionValuesChange")

    @property
    def feature_importances_(self):
        """
        Returns normalized feature importance values for CatBoost, ensuring
        the sum of all importances equals 1, making it comparable to
        feature importances from XGBoost, RandomForest, and DecisionTree.
        """
        importances = self.model.get_feature_importance(type="PredictionValuesChange")

        # 避免除零错误
        total_importance = sum(importances)
        if total_importance > 0:
            return importances / total_importance  # 归一化，使总和为 1
        else:
            return importances  # 如果全是 0，就直接返回
```



```
from sklearn.tree import DecisionTreeRegressor

class DTRegression:
    """
    Decision Tree with optional ccp_alpha (cost-complexity pruning).
    """
    def __init__(self, max_depth=None, random_state=42, ccp_alpha=0.0):
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=random_state,
            ccp_alpha=ccp_alpha
        )

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_
```



```
"""
models/model_rf.py

A Random Forest regressor wrapper using scikit-learn.
"""

from sklearn.ensemble import RandomForestRegressor

class RFRegression:
    """
    A simple regression model using RandomForest from scikit-learn.
    """

    def __init__(self, n_estimators=100, max_depth=10, random_state=42,
                 ccp_alpha=0.0, min_samples_leaf=1):
        """
               :param n_estimators: number of trees in the forest
               :param max_depth: maximum depth of the trees
               :param random_state: seed for reproducibility
               """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            ccp_alpha=ccp_alpha,
            min_samples_leaf=min_samples_leaf
        )

    def fit(self, X, Y):
        """
        Train the Random Forest model.
        :param X: training inputs, shape (N, input_dim)
        :param Y: training targets, shape (N, output_dim)
        """
        self.model.fit(X, Y)

    def predict(self, X):
        """
        Predict using the trained Random Forest model.
        :param X: inputs, shape (N, input_dim)
        :return: predictions, shape (N, output_dim)
        """
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        """
        方便外部直接访问 => rf_model.feature_importances_
        """
        return self.model.feature_importances_
```



```
"""
models/model_xgb.py

A simple XGBoost regressor wrapper using the scikit-learn API.
"""

from xgboost import XGBRegressor

class XGBRegression:
    """
    XGBoost with reg_alpha, reg_lambda for L1/L2 regularization.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 random_state=42, reg_alpha=0.0, reg_lambda=1.0):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            verbosity=0,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda
        )

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_
```



```
"""
trainers/train_sklearn.py

Contains a simple 'fit' function for sklearn-based models,
as they do not require an epoch-based training loop.
"""

def train_sklearn_model(model, X_train, Y_train):
    """
    Train a sklearn model (e.g., RandomForestRegressor).
    :param model: the sklearn model instance
    :param X_train: shape (N, input_dim)
    :param Y_train: shape (N, output_dim)
    :return: the trained model
    """
    model.fit(X_train, Y_train)
    return model
```



```
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
                                 weight_decay=0.0,
                                 checkpoint_path=None,
                                 log_interval=5,
                                 early_stopping=False,
                                 patience=5,
                                 optimizer_name="Adam"):
    """
    Train a PyTorch model using DataLoader, with optional Early Stopping & Dropout.

    :param model: 你的PyTorch模型 (继承nn.Module)
    :param train_dataset: 训练集 (Dataset类型)
    :param val_dataset: 验证集 (Dataset类型)
    :param loss_fn: 损失函数 (callable)
    :param epochs: 训练轮数
    :param batch_size: batch大小
    :param lr: 学习率
    :param weight_decay: L2正则系数
    :param checkpoint_path: 如果不为空，则在验证集loss改善时保存模型
    :param log_interval: 打印日志的间隔 (单位: epoch)
    :param early_stopping: 是否启用早停
    :param patience: 在多少个epoch内val_loss无改善，则停止
    :param optimizer_name: 指定优化器名称 (如"Adam", "AdamW")

    :return:
        - model: 训练后的模型(已恢复best_state)
        - train_losses: 每个epoch的训练集loss
        - val_losses: 每个epoch的验证集loss
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ----- 这里根据 optimizer_name 来选择不同的优化器 -----
    optimizer_name = optimizer_name.strip().lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"[INFO] Using AdamW optimizer (lr={lr}, weight_decay={weight_decay})")
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"[INFO] Using Adam optimizer (lr={lr}, weight_decay={weight_decay})")
    else:
        raise ValueError(f"Unknown optimizer_name='{optimizer_name}', only support 'Adam' or 'AdamW' so far.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    train_losses = []
    val_losses = []

    no_improve_epochs = 0  # 用于记录没有改善的轮数

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        batch_loss_list = []
        for (X_batch, Y_batch) in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            pred_batch = model(X_batch)
            loss_batch = loss_fn(pred_batch, Y_batch)

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            batch_loss_list.append(loss_batch.item())

        train_loss = float(np.mean(batch_loss_list))

        # ---- Validation ----
        model.eval()
        val_loss_list = []
        with torch.no_grad():
            for (X_val_b, Y_val_b) in val_loader:
                X_val_b = X_val_b.to(device)
                Y_val_b = Y_val_b.to(device)
                val_pred_b = model(X_val_b)
                loss_b = loss_fn(val_pred_b, Y_val_b)
                val_loss_list.append(loss_b.item())

        val_loss = float(np.mean(val_loss_list))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % log_interval == 0:
            print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Check if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve_epochs = 0
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            no_improve_epochs += 1

        # Early stopping check
        if early_stopping and no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch}. "
                  f"Best val_loss={best_val_loss:.6f} at epoch {best_epoch}.")
            break

    # 恢复最优状态
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model with val_loss={best_val_loss:.6f} from epoch {best_epoch}.")

    return model, train_losses, val_losses
```



```
"""
main.py

需求:
1) 同一个脚本支持:
   - K折 (可选, use_k_fold=true) => fold_{i}/train, fold_{i}/valid
   - 单次拆分 => full/train, full/valid
2) 模型创建统一在 create_model_by_type(...) 里
   - 对深度学习(ANN) => 需要 input_dim => 在split后确定
   - 其他(RF,DT,XGB,CatBoost) 不需要 input_dim
3) 训练:
   - 深度学习 => train_torch_model_dataloader (可含 dropout, early-stopping)
   - 机器学习 => train_sklearn_model
4) 可视化:
   - 训练/验证都画 MSE散点, MAE散点, residual_hist, residual_kde
   - 特征重要度(仅RF,DT,XGB,CatBoost)
5) 最后输出4张 3-metrics对比图:
   - KFold: train => three_metrics_horizontal_kfold_train.jpg
            val   => three_metrics_horizontal_kfold_val.jpg
   - Single-split: train => three_metrics_horizontal_train.jpg
                   val   => three_metrics_horizontal_val.jpg

新增: 在 K折 和 单次拆分 后, 若 config["evaluation"]["save_models_evaluation_bar"] 为 True，
     则计算并绘制模型“过拟合可视化”条形图:
       - overfitting_kfold.jpg  (K折)
       - overfitting_single.jpg (单次拆分)

【本版本额外新增】:
A) 在画完相关性图后, 若 config["evaluation"]["save_data_analysis_plots"] 为 True,
   调用 plot_advanced_data_analysis(...) 对原始df做更多数据可视化(如KDE,散点,盒须图等).
B) 在单次拆分后, 对每个模型保存:
   - 对ANN: best_ann.pt (已在train里保存), 其余 => trained_model.pkl
   - scaler_x_{mtype}.pkl, scaler_y_{mtype}.pkl
   - x_col_names.npy, y_col_names.npy
   都存放到 ./models/<model_type>/ 下, 供 inference 阶段使用.
"""

import yaml
import os
import numpy as np
import torch

# ============ 数据处理 =============
from data_preprocessing.data_loader import load_dataset, load_raw_data_for_correlation
from data_preprocessing.data_split import split_data, kfold_split_data
from data_preprocessing.my_dataset import MyDataset
from data_preprocessing.scaler_utils import (
    standardize_data, inverse_transform_output
)

# ============ 所有模型 =============
from models.model_rf import RFRegression
from models.model_dt import DTRegression
from models.model_catboost import CatBoostRegression
from models.model_xgb import XGBRegression
from models.model_ann import ANNRegression
# from models.model_gnn import GNNRegression  # 若有 GNN，也可导入

# ============ 损失 & 训练脚本 =============
from losses.torch_losses import get_torch_loss_fn
from trainers.train_torch import train_torch_model_dataloader
from trainers.train_sklearn import train_sklearn_model

# ============ 指标 & 可视化 =============
from evaluation.metrics import compute_regression_metrics
from evaluation.visualization import (
    plot_loss_curve,
    plot_scatter_3d_outputs_mse,
    plot_scatter_3d_outputs_mae,
    plot_residual_histogram,
    plot_residual_kde,
    plot_correlation_heatmap,
    plot_rf_feature_importance_bar,
    plot_three_metrics_horizontal,
    plot_overfitting_horizontal,
    # ★ 新增
    plot_advanced_data_analysis  # 你在 visualization.py 中新增的综合可视化函数
)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def create_model_by_type(model_type, config, random_seed=42, input_dim=None):
    """
    在此统一创建所有模型.
    对ANN => 若 input_dim 不为 None, 则覆盖 config 里写的.
    其余模型 => 不需要 input_dim.
    """
    if model_type == "ANN":
        ann_cfg = config["model"]["ann_params"]
        actual_dim = input_dim if input_dim is not None else ann_cfg["input_dim"]

        # 读取 ANN 专属随机种子; 若没写就用个默认值
        ann_random_seed = ann_cfg.get("random_seed", 42)

        model = ANNRegression(
            input_dim=actual_dim,
            output_dim=ann_cfg["output_dim"],
            hidden_dims=ann_cfg["hidden_dims"],
            dropout=ann_cfg.get("dropout", 0.0),
            activation=ann_cfg.get("activation", "ReLU"),
            random_seed=ann_random_seed
        )
        return model

    elif model_type == "RF":
        rf_cfg = config["model"]["rf_params"]
        return RFRegression(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            random_state=rf_cfg["random_state"],
            ccp_alpha=rf_cfg.get("ccp_alpha", 0.0),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 1)
        )
    elif model_type == "DT":
        dt_cfg = config["model"]["dt_params"]
        return DTRegression(
            max_depth=dt_cfg["max_depth"],
            random_state=dt_cfg["random_state"],
            ccp_alpha=dt_cfg.get("ccp_alpha", 0.0)
        )
    elif model_type == "CatBoost":
        cat_cfg = config["model"]["catboost_params"]
        return CatBoostRegression(
            iterations=cat_cfg["iterations"],
            learning_rate=cat_cfg["learning_rate"],
            depth=cat_cfg["depth"],
            random_seed=cat_cfg["random_seed"],
            l2_leaf_reg=cat_cfg.get("l2_leaf_reg", 3.0)
        )
    elif model_type == "XGB":
        xgb_cfg = config["model"]["xgb_params"]
        return XGBRegression(
            n_estimators=xgb_cfg["n_estimators"],
            learning_rate=xgb_cfg["learning_rate"],
            max_depth=xgb_cfg["max_depth"],
            random_state=xgb_cfg["random_seed"],
            reg_alpha=xgb_cfg.get("reg_alpha", 0.0),
            reg_lambda=xgb_cfg.get("reg_lambda", 1.0)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    csv_path = config["data"]["path"]
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]

    # e.g. "Dataset_XXXX/DataCorrelation"
    data_corr_rel = os.path.join(csv_name, "DataCorrelation")
    # e.g. "Dataset_XXXX/model_comparison"
    model_comp_rel= os.path.join(csv_name, "model_comparison")
    ensure_dir(os.path.join("./evaluation/figures", data_corr_rel))
    ensure_dir(os.path.join("./evaluation/figures", model_comp_rel))

    # =========== 1) Load data ===========
    X, Y, numeric_cols_idx, x_col_names, y_col_names = load_dataset(csv_path)

    # =========== 2) (可选)相关性图 ===========
    if config["evaluation"].get("save_correlation", False):
        df_raw_14 = load_raw_data_for_correlation(csv_path, drop_nan=True)
        raw_cols = df_raw_14.columns.tolist()
        numeric_14 = df_raw_14.select_dtypes(include=[np.number]).columns.tolist()
        numeric_idx_14 = [raw_cols.index(c) for c in numeric_14]

        fn1 = os.path.join(data_corr_rel, "correlation_heatmap.jpg")
        plot_correlation_heatmap(df_raw_14.values, col_names=raw_cols,
                                 numeric_cols_idx=numeric_idx_14, filename=fn1)

        fn2 = os.path.join(data_corr_rel, "correlation_heatmap_one_hot.jpg")
        plot_correlation_heatmap(X, col_names=x_col_names,
                                 numeric_cols_idx=range(X.shape[1]),
                                 filename=fn2)

    # =========== 2.5) (可选)数据分析可视化(新的) ===========
    # 在上面已经读取了 df_raw_14(未OneHot, 原生14列),
    # 若 config["evaluation"]["save_data_analysis_plots"] => True
    # 就调用 plot_advanced_data_analysis(...) 做更多图(如KDE,散点,盒图)
    if config["evaluation"].get("save_data_analysis_plots", False):
        if 'df_raw_14' in locals():
            # 把图像输出依旧放到 DataCorrelation 下
            from evaluation.visualization import plot_advanced_data_analysis
            out_dir = os.path.join("./evaluation/figures", data_corr_rel)
            plot_advanced_data_analysis(df_raw_14, out_dir=out_dir)
        else:
            print("[WARNING] df_raw_14 not loaded, skip advanced data plots.")


    # =========== 3) K折 or 单次拆分 参数 ===========
    use_k_fold = config["data"].get("use_k_fold", False)
    k_folds    = config["data"].get("k_folds", 5)
    random_seed= config["data"].get("random_seed", 42)
    model_types= config["model"]["types"]

    # 通过 config 里的 "save_models_evaluation_bar" 来决定是否绘制过拟合可视化
    save_overfit_bar = config["evaluation"].get("save_models_evaluation_bar", False)

    # -----------------------------------------------------------------
    # PART A: K-FOLD
    # -----------------------------------------------------------------
    kfold_train_metrics = {}
    kfold_val_metrics   = {}

    if use_k_fold and k_folds > 1:
        print(f"[INFO] K-Fold={k_folds}, Models={model_types}")

        for mtype in model_types:
            print(f"\n--- K-Fold for {mtype} ---")

            fold_train_list= []
            fold_val_list  = []

            model_sub_rel = os.path.join(model_comp_rel, mtype)
            ensure_dir(os.path.join("./evaluation/figures", model_sub_rel))

            # 调用 kfold_split_data 产生各折
            folds_data = kfold_split_data(
                X, Y,
                n_splits=k_folds,
                random_state=random_seed,
                shuffle=True
            )
            for fold_i, (X_tr, X_va, Y_tr, Y_va) in enumerate(folds_data, start=1):
                fold_dir_rel = os.path.join(model_sub_rel, f"fold_{fold_i}")
                ensure_dir(os.path.join("./evaluation/figures", fold_dir_rel))

                train_sub_rel= os.path.join(fold_dir_rel, "train")
                valid_sub_rel= os.path.join(fold_dir_rel, "valid")
                ensure_dir(os.path.join("./evaluation/figures", train_sub_rel))
                ensure_dir(os.path.join("./evaluation/figures", valid_sub_rel))

                # 标准化
                (X_tr_s, X_va_s, sx), (Y_tr_s, Y_va_s, sy) = standardize_data(
                    X_tr, X_va, Y_tr, Y_va,
                    do_input=config["preprocessing"]["standardize_input"],
                    do_output=config["preprocessing"]["standardize_output"],
                    numeric_cols_idx=numeric_cols_idx
                )

                # 若是ANN，需要动态获取 input_dim
                actual_input_dim= X_tr_s.shape[1]

                # ========== 创建模型 =============
                model= create_model_by_type(
                    model_type=mtype,
                    config=config,
                    random_seed=random_seed,
                    input_dim=actual_input_dim
                )

                # ========== 训练 =============
                if mtype in ["ANN"]:
                    # PyTorch
                    loss_fn= get_torch_loss_fn(config["loss"]["type"])
                    train_ds= MyDataset(X_tr_s, Y_tr_s)
                    val_ds=   MyDataset(X_va_s, Y_va_s)

                    if mtype=="ANN":
                        ann_cfg= config["model"]["ann_params"]
                        lr_= float(ann_cfg["learning_rate"])
                        bs_= ann_cfg["batch_size"]
                        ep_= ann_cfg["epochs"]
                        wdecay= float(ann_cfg.get("weight_decay", 0.0))
                        ckp= None  # K折一般不保存checkpoint
                        opt_name = ann_cfg.get("optimizer", "Adam")
                        do_early_stopping = ann_cfg.get("early_stopping", False)
                        patience_ = ann_cfg.get("patience", 5)

                    model, _, _= train_torch_model_dataloader(
                        model, train_ds, val_ds,
                        loss_fn=loss_fn,
                        epochs=ep_,
                        batch_size=bs_,
                        lr=lr_,
                        weight_decay=wdecay,
                        checkpoint_path=ckp,
                        log_interval=config["training"]["log_interval"],
                        early_stopping=do_early_stopping,
                        patience=patience_,
                        optimizer_name=opt_name
                    )

                    model.to("cpu")
                    with torch.no_grad():
                        train_pred= model(torch.tensor(X_tr_s, dtype=torch.float32)).cpu().numpy()
                        val_pred=   model(torch.tensor(X_va_s, dtype=torch.float32)).cpu().numpy()

                else:
                    # Sklearn
                    model= train_sklearn_model(model, X_tr_s, Y_tr_s)
                    train_pred= model.predict(X_tr_s)
                    val_pred=   model.predict(X_va_s)

                # 若输出也标准化，则反转
                if config["preprocessing"]["standardize_output"]:
                    train_pred= inverse_transform_output(train_pred, sy)
                    val_pred=   inverse_transform_output(val_pred, sy)

                # 计算指标
                train_m= compute_regression_metrics(Y_tr, train_pred)
                val_m=   compute_regression_metrics(Y_va, val_pred)
                print(f"   Fold{fold_i} => train={train_m}, valid={val_m}")

                fold_train_list.append(train_m)
                fold_val_list.append(val_m)

                # ========== 可视化 (train+val) => 例如MSE散点、MAE散点等 ==========
                if config["evaluation"].get("save_scatter_mse_plot", False):
                    out_mse_tr= os.path.join(train_sub_rel, f"{mtype}_fold{fold_i}_mse_scatter_train.jpg")
                    plot_scatter_3d_outputs_mse(Y_tr, train_pred, None, filename=out_mse_tr)

                    out_mse_val= os.path.join(valid_sub_rel, f"{mtype}_fold{fold_i}_mse_scatter.jpg")
                    plot_scatter_3d_outputs_mse(Y_va, val_pred, None, filename=out_mse_val)

            # -- 所有fold结束 => 计算平均指标
            train_mse= [fm["MSE"] for fm in fold_train_list]
            train_mae= [fm["MAE"] for fm in fold_train_list]
            train_r2 = [fm["R2"]  for fm in fold_train_list]
            avg_train_m= {
                "MSE": np.mean(train_mse),
                "MAE": np.mean(train_mae),
                "R2" : np.mean(train_r2)
            }

            val_mse= [fm["MSE"] for fm in fold_val_list]
            val_mae= [fm["MAE"] for fm in fold_val_list]
            val_r2 = [fm["R2"]  for fm in fold_val_list]
            avg_val_m= {
                "MSE": np.mean(val_mse),
                "MAE": np.mean(val_mae),
                "R2" : np.mean(val_r2)
            }
            print(f"   => K-Fold average: train={avg_train_m}, valid={avg_val_m}")

            # 保存到kfold_train_metrics, kfold_val_metrics
            kfold_train_metrics[mtype]= avg_train_m
            kfold_val_metrics[mtype]  = avg_val_m

        # 画 3 指标条形图 (train, val)
        if len(kfold_train_metrics) > 0:
            out_kf_tr = os.path.join(data_corr_rel, "three_metrics_horizontal_kfold_train.jpg")
            plot_three_metrics_horizontal(kfold_train_metrics, save_name=out_kf_tr)
        if len(kfold_val_metrics) > 0:
            out_kf_val= os.path.join(data_corr_rel, "three_metrics_horizontal_kfold_val.jpg")
            plot_three_metrics_horizontal(kfold_val_metrics, save_name=out_kf_val)

        # =========== 如果配置中 "save_models_evaluation_bar"=True => 过拟合可视化 (K-Fold) ============
        if save_overfit_bar:
            # 需要训练集、验证集K折均值
            if len(kfold_train_metrics)>0 and len(kfold_val_metrics)>0:
                overfit_kf = {}
                for m in kfold_train_metrics:
                    trm = kfold_train_metrics[m]
                    vam = kfold_val_metrics[m]

                    # MSE_ratio
                    if trm["MSE"] == 0:
                        ms_ratio = float("inf")
                    else:
                        ms_ratio = vam["MSE"] / trm["MSE"]

                    # R2_diff
                    r2_diff = trm["R2"] - vam["R2"]

                    overfit_kf[m] = {
                        "MSE_ratio": ms_ratio,
                        "R2_diff":   r2_diff
                    }

                out_kf_of = os.path.join(data_corr_rel, "overfitting_kfold.jpg")
                plot_overfitting_horizontal(overfit_kf, save_name=out_kf_of)

    # -----------------------------------------------------------------
    # PART B: 单次拆分
    # -----------------------------------------------------------------
    single_train_dict= {}
    single_val_dict= {}

    print("\n[INFO] Single-split => produce train/val images.")
    # 拆分训练/验证
    X_train, X_val, Y_train, Y_val= split_data(
        X, Y,
        test_size=config["data"]["test_size"],
        random_state=random_seed
    )

    (X_train_s, X_val_s, sx), (Y_train_s, Y_val_s, sy)= standardize_data(
        X_train, X_val, Y_train, Y_val,
        do_input=config["preprocessing"]["standardize_input"],
        do_output=config["preprocessing"]["standardize_output"],
        numeric_cols_idx=numeric_cols_idx
    )

    dim_input= X_train_s.shape[1]

    for mtype in model_types:
        print(f"\n[Train model: {mtype}]")

        model_sub_rel= os.path.join(model_comp_rel, mtype)
        ensure_dir(os.path.join("./evaluation/figures", model_sub_rel))

        full_train_rel= os.path.join(model_sub_rel, "full", "train")
        full_val_rel=   os.path.join(model_sub_rel, "full", "valid")
        ensure_dir(os.path.join("./evaluation/figures", full_train_rel))
        ensure_dir(os.path.join("./evaluation/figures", full_val_rel))

        model= create_model_by_type(
            model_type=mtype,
            config=config,
            random_seed=random_seed,
            input_dim=dim_input
        )

        if mtype in ["ANN"]:
            loss_fn= get_torch_loss_fn(config["loss"]["type"])

            if mtype=="ANN":
                ann_cfg= config["model"]["ann_params"]
                lr_= float(ann_cfg["learning_rate"])
                bs_= ann_cfg["batch_size"]
                ep_= ann_cfg["epochs"]
                wdecay= float(ann_cfg.get("weight_decay", 0.0))
                ckp= ann_cfg["checkpoint_path"]
                opt_name= ann_cfg.get("optimizer", "Adam")
                do_early_stopping = ann_cfg.get("early_stopping", False)
                patience_ = ann_cfg.get("patience", 5)

            train_ds= MyDataset(X_train_s, Y_train_s)
            val_ds=   MyDataset(X_val_s,   Y_val_s)

            model, train_losses, val_losses= train_torch_model_dataloader(
                model,
                train_ds,
                val_ds,
                loss_fn=loss_fn,
                epochs=ep_,
                batch_size=bs_,
                lr=lr_,
                weight_decay=wdecay,
                checkpoint_path=ckp,  # e.g. ./models/ANN/best_ann.pt
                log_interval=config["training"]["log_interval"],
                early_stopping=do_early_stopping,
                patience=patience_,
                optimizer_name=opt_name
            )
            model.to("cpu")
            with torch.no_grad():
                train_pred= model(torch.tensor(X_train_s, dtype=torch.float32)).cpu().numpy()
                val_pred=   model(torch.tensor(X_val_s,   dtype=torch.float32)).cpu().numpy()

            if config["evaluation"].get("save_loss_curve", False):
                out_lc= os.path.join(model_sub_rel, f"{mtype}_loss_curve.jpg")
                plot_loss_curve(train_losses, val_losses, filename=out_lc)

        else:
            # Sklearn
            model= train_sklearn_model(model, X_train_s, Y_train_s)
            train_pred= model.predict(X_train_s)
            val_pred=   model.predict(X_val_s)

        # 若输出也做标准化 => 反转
        if config["preprocessing"]["standardize_output"]:
            train_pred= inverse_transform_output(train_pred, sy)
            val_pred=   inverse_transform_output(val_pred,   sy)

        train_m= compute_regression_metrics(Y_train, train_pred)
        val_m  = compute_regression_metrics(Y_val,   val_pred)
        print(f"  [Train Metrics] => {train_m}")
        print(f"  [Valid Metrics] => {val_m}")

        single_train_dict[mtype]= train_m
        single_val_dict[mtype]  = val_m

        # =========== 可视化 => train+val ================
        if config["evaluation"].get("save_scatter_mse_plot",False):
            out_mse_tr= os.path.join(full_train_rel, f"{mtype}_mse_scatter_train.jpg")
            plot_scatter_3d_outputs_mse(Y_train, train_pred, y_labels=y_col_names, filename=out_mse_tr)

            out_mse_val= os.path.join(full_val_rel, f"{mtype}_mse_scatter.jpg")
            plot_scatter_3d_outputs_mse(Y_val, val_pred, y_labels=y_col_names, filename=out_mse_val)

        if config["evaluation"].get("save_scatter_mae_plot",False):
            out_mae_tr= os.path.join(full_train_rel, f"{mtype}_mae_scatter_train.jpg")
            plot_scatter_3d_outputs_mae(Y_train, train_pred, y_labels=y_col_names, filename=out_mae_tr)

            out_mae_val= os.path.join(full_val_rel, f"{mtype}_mae_scatter.jpg")
            plot_scatter_3d_outputs_mae(Y_val, val_pred, y_labels=y_col_names, filename=out_mae_val)

        if config["evaluation"].get("save_residual_hist",False):
            out_hist_tr= os.path.join(full_train_rel, f"{mtype}_residual_hist_train.jpg")
            plot_residual_histogram(Y_train, train_pred, y_labels=y_col_names, filename=out_hist_tr)

            out_hist_val= os.path.join(full_val_rel, f"{mtype}_residual_hist.jpg")
            plot_residual_histogram(Y_val, val_pred, y_labels=y_col_names, filename=out_hist_val)

        if config["evaluation"].get("save_residual_kde",False):
            out_kde_tr= os.path.join(full_train_rel, f"{mtype}_residual_kde_train.jpg")
            plot_residual_kde(Y_train, train_pred, y_labels=y_col_names, filename=out_kde_tr)

            out_kde_val= os.path.join(full_val_rel, f"{mtype}_residual_kde.jpg")
            plot_residual_kde(Y_val, val_pred, y_labels=y_col_names, filename=out_kde_val)

        # 特征重要度 => 仅对 (RF,DT,XGB,CatBoost)
        if config["evaluation"].get("save_feature_importance_bar", False):
            if hasattr(model, "feature_importances_"):
                fi_out = os.path.join(model_sub_rel, f"{mtype}_feature_importance.jpg")
                plot_rf_feature_importance_bar(model, x_col_names, filename=fi_out)
            else:
                if hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
                    fi_out = os.path.join(model_sub_rel, f"{mtype}_feature_importance.jpg")
                    plot_rf_feature_importance_bar(model.model, x_col_names, filename=fi_out)

        # ============== 新增：保存模型 + scaler + colnames ==============
        from data_preprocessing.scaler_utils import save_scaler
        import joblib

        model_dir = os.path.join("./models", mtype)
        ensure_dir(model_dir)

        # scaler_x_{mtype}.pkl / scaler_y_{mtype}.pkl
        sx_path = os.path.join(model_dir, f"scaler_x_{mtype}.pkl")
        sy_path = os.path.join(model_dir, f"scaler_y_{mtype}.pkl")
        save_scaler(sx, sx_path)
        save_scaler(sy, sy_path)

        # 保存列名 => x_col_names.npy, y_col_names.npy
        np.save(os.path.join(model_dir, "x_col_names.npy"), x_col_names)
        np.save(os.path.join(model_dir, "y_col_names.npy"), y_col_names)

        if mtype=="ANN":
            # best_ann.pt 在train过程里已保存 => ann_cfg["checkpoint_path"]
            print(f"[INFO] ANN => checkpoint saved at: {ckp}")
        else:
            # sklearn/catboost/xgb => joblib
            out_pkl = os.path.join(model_dir, "trained_model.pkl")
            joblib.dump(model, out_pkl)
            print(f"[INFO] Saved {mtype} => {out_pkl}")


    # => 画 3 metrics => train
    if len(single_train_dict) > 0:
        out_3_train= os.path.join(data_corr_rel, "three_metrics_horizontal_train.jpg")
        plot_three_metrics_horizontal(single_train_dict, save_name=out_3_train)

    # => val
    if len(single_val_dict) > 0:
        out_3_val= os.path.join(data_corr_rel, "three_metrics_horizontal_val.jpg")
        plot_three_metrics_horizontal(single_val_dict, save_name=out_3_val)

    # ============ 如果 "save_models_evaluation_bar"=True => Overfitting可视化 (Single-split) ============
    if save_overfit_bar:
        if len(single_train_dict) > 0 and len(single_val_dict) > 0:
            overfit_single= {}
            for m in single_train_dict:
                trm= single_train_dict[m]
                vam= single_val_dict[m]
                # MSE_ratio = valMSE / trainMSE
                if trm["MSE"] == 0:
                    ms_ratio= float("inf")
                else:
                    ms_ratio= vam["MSE"] / trm["MSE"]
                # R2_diff   = (Train R² - Val R²)
                r2_diff= trm["R2"] - vam["R2"]

                overfit_single[m]= {
                    "MSE_ratio": ms_ratio,
                    "R2_diff":   r2_diff
                }

            out_overfit_single= os.path.join(data_corr_rel, "overfitting_single.jpg")
            plot_overfitting_horizontal(overfit_single, save_name=out_overfit_single)

    print("\n[INFO] Single-split done.")
    print("Train metrics =>", single_train_dict)
    print("Val   metrics =>", single_val_dict)


if __name__ == "__main__":
    main()
```



```
"""
inference.py

需求:
1) 在 config.yaml 的 inference.models 中指定要推理的模型
2) 加载每个模型(ANN => best_ann.pt, 其余 => trained_model.pkl) + scaler + colnames
3) 生成 potential,catalyst_size 网格 + multi-cat onehot => 多次预测 => 取均值
4) 画 4 张 heatmap => <csv_name>/inference/<model_type>/mimo_heatmap_*.jpg
"""

import os
import yaml
import numpy as np
import torch
import joblib
from data_preprocessing.data_loader import load_dataset
from data_preprocessing.scaler_utils import load_scaler, inverse_transform_output
from evaluation.visualization import plot_2d_mimo_heatmaps
from models.model_ann import ANNRegression
from models.model_rf import RFRegression
from models.model_dt import DTRegression
from models.model_catboost import CatBoostRegression
from models.model_xgb import XGBRegression

def load_inference_model(model_type, config):
    """
    在 ./models/<model_type>/ 下加载:
      - ANN => best_ann.pt
      - 其余 => trained_model.pkl
    """
    model_dir = os.path.join("./models", model_type)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"[ERROR] Directory not found => {model_dir}")

    if model_type=="ANN":
        ann_cfg = config["model"]["ann_params"]
        # 构造同样维度ANN
        net = ANNRegression(
            input_dim=ann_cfg["input_dim"],
            output_dim=ann_cfg["output_dim"],
            hidden_dims=ann_cfg["hidden_dims"],
            dropout=ann_cfg.get("dropout",0.0),
            activation=ann_cfg.get("activation","ReLU"),
            random_seed=ann_cfg.get("random_seed",42)
        )
        ckpt_path = os.path.join(model_dir, "best_ann.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[ERROR] {ckpt_path} not found.")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(state_dict)
        net.eval()
        return net
    else:
        pkl_path = os.path.join(model_dir, "trained_model.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"[ERROR] {pkl_path} not found.")
        model = joblib.load(pkl_path)
        return model

def identify_categorical_groups(x_col_names):
    prefix_map = {}
    for i, cname in enumerate(x_col_names):
        if "_" in cname:
            prefix = cname.split("_")[0]
            prefix_map.setdefault(prefix, []).append(i)
    cat_groups = []
    for pref, idxs in prefix_map.items():
        if len(idxs)>=2:
            cat_groups.append(sorted(idxs))
    return cat_groups

def enumerate_cat_combinations(cat_groups, max_combos=20):
    from itertools import product
    all_combos = list(product(*cat_groups))
    if len(all_combos) > max_combos:
        print(f"[WARN] total combos={len(all_combos)}, truncated to {max_combos}.")
        all_combos = all_combos[:max_combos]
    return all_combos

def run_inference_for_one_model(mtype, config):
    """
    1) 加载模型 & scaler & colnames
    2) 构造 (potential,catalyst_size) 网格
    3) 多分类组合 => 取均值
    4) inverse_transform => 画热力图
    """
    model_dir = os.path.join("./models", mtype)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"[ERROR] No directory => {model_dir}")

    model = load_inference_model(mtype, config)
    sx_path = os.path.join(model_dir, f"scaler_x_{mtype}.pkl")
    sy_path = os.path.join(model_dir, f"scaler_y_{mtype}.pkl")
    if not os.path.exists(sx_path) or not os.path.exists(sy_path):
        raise FileNotFoundError(f"[ERROR] Missing scaler pkl => {sx_path} or {sy_path}")

    scaler_x = load_scaler(sx_path)
    scaler_y = load_scaler(sy_path)

    xcol_path = os.path.join(model_dir, "x_col_names.npy")
    ycol_path = os.path.join(model_dir, "y_col_names.npy")
    if not (os.path.exists(xcol_path) and os.path.exists(ycol_path)):
        raise FileNotFoundError("[ERROR] x_col_names.npy or y_col_names.npy missing.")
    x_col_names = list(np.load(xcol_path, allow_pickle=True))
    y_col_names = list(np.load(ycol_path, allow_pickle=True))

    # 读取 CSV, 获取 potential,catalyst_size min max
    csv_path = config["data"]["path"]
    X_, Y_, numeric_cols_idx, x_names_full, y_names_full = load_dataset(csv_path)
    # find indexes
    try:
        idx_pot = x_col_names.index("potential")
        idx_csize = x_col_names.index("catalyst_size")
    except ValueError:
        raise ValueError("[ERROR] 'potential' or 'catalyst_size' not in x_col_names. Check your data.")

    pot_min, pot_max = X_[:, idx_pot].min(), X_[:, idx_pot].max()
    csize_min, csize_max = X_[:, idx_csize].min(), X_[:, idx_csize].max()

    n_points = 50
    pot_vals = np.linspace(pot_min, pot_max, n_points)
    csize_vals = np.linspace(csize_min, csize_max, n_points)
    grid_pot, grid_csize = np.meshgrid(pot_vals, csize_vals)

    cat_groups = identify_categorical_groups(x_col_names)
    max_comb = config["inference"].get("max_combinations",20)
    combos = enumerate_cat_combinations(cat_groups, max_comb)

    # 其余连续变量 => X_.mean(axis=0)
    mean_vec = X_.mean(axis=0)

    H, W = grid_pot.shape
    output_dim = len(y_col_names)
    predictions = np.zeros((H,W,output_dim), dtype=np.float32)
    is_torch = (mtype=="ANN")

    for i in range(H):
        for j in range(W):
            base_x = mean_vec.copy()
            base_x[idx_pot]   = grid_pot[i,j]
            base_x[idx_csize] = grid_csize[i,j]

            combo_preds = np.zeros((len(combos), output_dim), dtype=np.float32)
            for c_idx, combo in enumerate(combos):
                x_vec = base_x.copy()
                # reset each cat group =>0
                for grp in cat_groups:
                    for colid in grp:
                        x_vec[colid] = 0
                # set 1
                for colid in combo:
                    x_vec[colid] = 1

                x_2d = x_vec.reshape(1,-1)
                numeric_part = x_2d[:, numeric_cols_idx]
                scaled_num = scaler_x.transform(numeric_part)
                x_2d[:, numeric_cols_idx] = scaled_num
                if is_torch:
                    import torch
                    inp_t = torch.tensor(x_2d, dtype=torch.float32)
                    with torch.no_grad():
                        out_t = model(inp_t)
                    out_np = out_t.cpu().numpy()[0]
                else:
                    out_np = model.predict(x_2d)[0]
                combo_preds[c_idx,:] = out_np
            avg_pred = combo_preds.mean(axis=0)
            predictions[i,j,:] = avg_pred

    # inverse transform y
    phw = predictions.reshape(-1, output_dim)
    phw_inv = inverse_transform_output(phw, scaler_y)
    predictions = phw_inv.reshape(H, W, output_dim)

    # 保存热力图 => <csv_name>/inference/<mtype>/
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.join(csv_name, "inference", mtype)
    os.makedirs(out_dir, exist_ok=True)
    plot_2d_mimo_heatmaps(grid_pot, grid_csize, predictions, out_labels=y_col_names, out_dir=out_dir)
    print(f"[INFO] Inference => {mtype}, results saved to {out_dir}")


def main():
    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 读取 inference.models
    inf_models = config["inference"].get("models", [])
    if len(inf_models)==0:
        print("[INFO] No inference models specified, exit.")
        return

    for mt in inf_models:
        run_inference_for_one_model(mt, config)


if __name__=="__main__":
    main()
```



请帮我处理好所有的问题给我一个一次性可以运行的项目,给我所有修改后的完整代码.

------

## **1. 背景**

在我的机器学习/深度学习任务中，我训练了一个 **多输入-多输出（MIMO）回归模型**，其中：

- 输入变量

    ：

    - **连续变量**（如 `potential`、`catalyst size`、`temperature`）。
    - **类别变量**（如 `catalyst type`，采用 one-hot 编码）。

- 输出变量

    ：

    - 4 个预测目标（如不同产物的产量）。

- 数据预处理

    ：

    - 连续变量进行了标准化（均值、标准差）。
    - 类别变量进行了 one-hot 编码。

在推理阶段，我希望分析 **`potential` 和 `catalyst size` 这两个关键变量** 对四个预测目标的影响，并生成 **热力图** 来直观展示它们在不同区域的变化趋势。

------

## **2. 目的**

我希望通过 **后处理分析**：

1. **分析 `potential` 和 `catalyst size` 对输出变量的影响趋势**，确定在哪个区域某个产物的产量最高。
2. **在绘制热力图时，类别变量（`catalyst type`）取多个类别的均值**，以消除类别变量的干扰，得到更加全面的趋势分析。
3. **保证训练和推理时的数据一致性**，确保数据预处理（标准化、one-hot 编码）在推理时仍然有效。

------

## **3. 方案**

### **（1）训练阶段**

- 存储所有输入变量的统计信息

    ：

    - 连续变量的 **最小值、最大值、均值、标准差**。
    - 类别变量的 **所有可能取值（用于 one-hot 编码）**。

- 存储训练好的模型参数

    ：

    - 深度学习：`model.pth`（PyTorch）或 `model.h5`（TensorFlow）。
    - 传统 ML：`xgboost.pkl`（XGBoost）或 `lightgbm.pkl`（LightGBM）。

### **（2）推理阶段**

1. **加载存储的标准化参数、one-hot 编码信息和训练好的模型**。

2. **生成 `potential` 和 `catalyst size` 的网格数据**（如 `50x50` 网格）。

3. 处理其他输入变量

    ：

    - **连续变量**（如 `temperature`）设为 **均值**。

    - 类别变量（`catalyst type`）的 one-hot 编码

        ：

        - 计算多个类别的均值：
            - 对每个 `potential, catalyst size` 组合，分别计算 **不同 `catalyst type` 下的预测值**。
            - 然后 **取多个类别预测值的均值**，生成最终的预测结果。

4. 进行模型推理

    ：

    - 预测形状为 `(50, 50, 4)`，即 **四个输出变量在网格上的分布**。

5. 绘制 4 张热力图

    ：

    - `X 轴 = potential`
    - `Y 轴 = catalyst size`
    - 颜色表示预测值

------

## **4. 预期结果**

- **清晰展示 `potential` 和 `catalyst size` 对输出变量的影响**，帮助优化参数选择。
- **通过取多个 `catalyst type` 预测值的均值**，减少类别变量带来的偏差，使分析更加合理。
- **可视化四个输出变量的变化趋势**，帮助发现最优参数区域。

------

## **5. 关键问题**

- 如何合理处理 one-hot 变量？
    - **采用多个类别预测值的均值**，避免类别变量影响整体趋势。
- 如何保证训练和推理时的数据一致性？
    - **存储并加载所有标准化参数、one-hot 映射、和模型参数**，保证推理时的输入数据匹配训练数据。
- 如何高效绘制多输出变量的热力图？
    - 采用 **`matplotlib.pcolormesh`**，生成 `(50x50)` 网格，分别绘制 4 张热力图。



你这里面有一个问题,我想要的inference并不是所谓的给出一串输入给出一串输出,而是我希望查看某两个数据(potential 和 catalyst_size.),请在函数型中给出变量选择的接口,并从我的dataloader.py中获取相应的参数名称的列表. 请包含相应的判断.输入对于我某一个输出的影响的热力图,因此我希望你在数据加载过程中针对不同的数据给出一些保存选项. 因为我们的输入一定是要完整的不能有空值,我的想法是给连续值数据输入平均值,给onehot数据,穷举所有可能,在其余值为平均值,两个自变量为当前选取点处急性多次计算,最后取平均值做出当前数据的绘图.因此涉及到一些保存和读取的问题1. 针对连续性数据, 保存其最大值,最小值,平均值. 2. 针对onehot类型,要记录下没一列中的不同onehot之间的关系,因为相同列中的onehot类型互斥,只能有一个激活,这样穷举就会更加切合实际.举个例子,比如我有两列onehot类型,第一列有3中onehot,第二列有2中onehot.此外我有4组连续型数据已经保存了读取了其最大值,最小值,平均值.我从中选取了两个自变量. 于是我读取这两个自变量的最大值最小值,确定了我的x_lim 和 y_lim.然后我开始进行等分绘图,在每一个绘图点上我有两个自变量的值. 另外两个连续性数据读取其平均值并输入到我的模型中, 然后onehot 我就有$C_3^1 * C_2^1 = 6$种不同的输入,把这六个输入代入进去算6次,然后除以6即可得到当前这一点的一个预测点,然后我就可以一口气绘制四个图片,因为我有我的四个输出. 当然这里面包含了标准化操作,以及相关的文件的保存和读取.需要你一并实现.这是我的inference的例子. 请修改.选择的话,给出参数进行选择,请结合data_loader.py和main.py进行整合修改,这是其中的一个图片,他应该是对应四个输出的四个子图.
下面介绍inference中的第二个图片输出,这个是关于我们的onehot的类别的,之前已经修改好了读取了所有的连续值的最大,最小,平均值.现在我要求你把连续值的相关的所有的输入全部按照平均值输入,然后对于onehot,上一版是onehot类多次穷举取平均,这一版输出我想考虑的是onehot的类型.也就是做一个混淆矩阵的heatmap,我要求你画出混淆矩阵,但是由于我们有四个输出,横纵坐标只包含我的onehot类别, 输入的连续类型变量直接用平均值代替, 混淆矩阵的显示结果为我的四个预测值,因此这个可能并不是传统的混淆矩阵计算.所以我会有四个输出,对应四种不同的colorbar,并且colorbar的名称就对应着我们的输出的y_label这个应该已经实现了.至于说在相同列中的onehot是绝对不可能有同时激活的情况,那么在此位置就给予空结果,可以脱离colorbar给成别的颜色.其余的计算正常计算.对于混淆矩阵中的每个单元格,里面应该是有4个分割子格子.请帮我由对角线分割成四个三角,这样更好看. colorbar就在右侧依次摆放即可.因为每次选取了两个onehot进行激活, 所以其余的onehot怎么选择希望你能帮我思考一下.或许按照之前的穷举平均方法会不会有点慢呢?但是似乎我的参数也没那么多,你思考下帮我解决.



以上的两个推断方式请按照函数的方式分别写出来,然后在脚本的末尾调用.



```
def plot_scatter_3d_outputs_mse(y_true, y_pred, y_labels=None, filename="scatter_3d_output_mse.jpg"):
    """
    Create subplots for multi-output regression (dimension = n_outputs).
    Each subplot:
      - X轴: True Y
      - Y轴: Predicted Y
      - Color: MSE (squared error) per sample for that output dimension.
      - 额外: 在标题处显示该维度的 R² 值.
    """
    ensure_fig_dir()

    if y_pred.ndim != 2:
        raise ValueError(f"y_pred shape expected (N, n_outputs), got {y_pred.shape}")
    n_samples, n_outputs = y_pred.shape

    fig, axes = plt.subplots(1, n_outputs, figsize=(4 * n_outputs, 4), squeeze=False)

    for i in range(n_outputs):
        # ========== 计算每个维度的MSE误差(每个点) ==========
        errors = (y_true[:, i] - y_pred[:, i]) ** 2

        # ========== 计算该维度的R² ==========
        r2_val = r2_score(y_true[:, i], y_pred[:, i])

        ax = axes[0, i]
        sc = ax.scatter(
            y_true[:, i],
            y_pred[:, i],
            c=errors,
            alpha=0.5,
            cmap='brg'
        )
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)

        # ========== 在子图标题处加 R² ==========
        if y_labels and i < len(y_labels):
            title_str = f"{y_labels[i]} (MSE colormap)\nR² = {r2_val:.3f}"
            ax.set_xlabel(f"True {y_labels[i]}")
            ax.set_ylabel(f"Pred {y_labels[i]}")
        else:
            title_str = f"Output {i} (MSE colormap)\nR² = {r2_val:.3f}"
            ax.set_xlabel("True")
            ax.set_ylabel("Pred")

        ax.set_title(title_str)

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Squared Error", fontweight='bold', fontsize=13)
        cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=700, format='jpg')
    plt.close()
```

第二个问题就是我发现你的catalyst_size_vs_product.jpg,  potential_vs_product_by_electrolyte.jpg  three_metrics_horizontal_val.jpg这几个图片全部现实Cols没有找到,全是空的,你这个输入参数决定类别的逻辑肯定出了问题,你好好看看我的数据夹在逻辑是怎么样的,很有可能你的读取列的方法和这个完全不匹配, 你可以修改这个数据加载文件,也可以修改main函数中的逻辑来实现相关列的调用.毕竟数据已经搞到了为什么不能读取呢.仔细思考改正.别忘了你的这个部分的绘图应该是没有标准化和onehot的数据哦:

```
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
```







kde_distribution.jpg           这个里面你似乎给所有的子图片用了相同的colorbar,我想要的是每个图片一个colorbar,用循环来进行绘制最好了,一个循环一个colorbar      下面这个代码的布局就还好,我说的只是colorbar的布局:

```
def plot_scatter_3d_outputs_mse(y_true, y_pred, y_labels=None, filename="scatter_3d_output_mse.jpg"):
    """
    Create subplots for multi-output regression (dimension = n_outputs).
    Each subplot:
      - X轴: True Y
      - Y轴: Predicted Y
      - Color: MSE (squared error) per sample for that output dimension.
      - 额外: 在标题处显示该维度的 R² 值.
    """
    ensure_fig_dir()

    if y_pred.ndim != 2:
        raise ValueError(f"y_pred shape expected (N, n_outputs), got {y_pred.shape}")
    n_samples, n_outputs = y_pred.shape

    fig, axes = plt.subplots(1, n_outputs, figsize=(4 * n_outputs, 4), squeeze=False)

    for i in range(n_outputs):
        # ========== 计算每个维度的MSE误差(每个点) ==========
        errors = (y_true[:, i] - y_pred[:, i]) ** 2

        # ========== 计算该维度的R² ==========
        r2_val = r2_score(y_true[:, i], y_pred[:, i])

        ax = axes[0, i]
        sc = ax.scatter(
            y_true[:, i],
            y_pred[:, i],
            c=errors,
            alpha=0.5,
            cmap='brg'
        )
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)

        # ========== 在子图标题处加 R² ==========
        if y_labels and i < len(y_labels):
            title_str = f"{y_labels[i]} (MSE colormap)\nR² = {r2_val:.3f}"
            ax.set_xlabel(f"True {y_labels[i]}")
            ax.set_ylabel(f"Pred {y_labels[i]}")
        else:
            title_str = f"Output {i} (MSE colormap)\nR² = {r2_val:.3f}"
            ax.set_xlabel("True")
            ax.set_ylabel("Pred")

        ax.set_title(title_str)

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Squared Error", fontweight='bold', fontsize=13)
        cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=700, format='jpg')
    plt.close()
```

请帮我综合考虑后彻底重写我的代码并实现上面的所有要求,请不要删除我的注释,可以重写注释,比如注释步骤顺序发生了改变等.

