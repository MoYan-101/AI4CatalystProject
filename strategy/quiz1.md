def mixed_correlation_matrix(X, col_names, numeric_cols_idx,

​                             method_numeric="pearson",

​                             dropna=True):

​    """

​    计算混合变量相关性矩阵: 数值-数值 => Pearson/Spearman,

​                       类别-类别 => Cramer's V,

​                       数值-类别 => Correlation Ratio.

​    """

​    n_features = X.shape[1]

​    if not isinstance(X, pd.DataFrame):

​        X = pd.DataFrame(X, columns=col_names)



​    numeric_set = set(numeric_cols_idx)

​    corr = np.zeros((n_features, n_features), dtype=float)



​    for i in range(n_features):

​        for j in range(n_features):

​            if i == j:

​                corr[i, j] = 1.0

​                continue

​            if i > j:

​                corr[i, j] = corr[j, i]

​                continue



​            col_i = col_names[i]

​            col_j = col_names[j]

​            data_i = X[col_i]

​            data_j = X[col_j]

​            if dropna:

​                valid = ~data_i.isna() & ~data_j.isna()

​                data_i = data_i[valid]

​                data_j = data_j[valid]



​            i_is_num = (i in numeric_set)

​            j_is_num = (j in numeric_set)



​            if i_is_num and j_is_num:

​                \# 数值-数值

​                if method_numeric.lower() == "pearson":

​                    r, _ = ss.pearsonr(data_i, data_j)

​                else:

​                    r, _ = ss.spearmanr(data_i, data_j)

​                corr[i, j] = r

​            elif (not i_is_num) and (not j_is_num):

​                \# 类别-类别

​                r = cramers_v(data_i, data_j)

​                corr[i, j] = r

​            else:

​                \# 数值-类别 => correlation_ratio

​                if i_is_num:

​                    cat_data, num_data = data_j, data_i

​                else:

​                    cat_data, num_data = data_i, data_j

​                r = correlation_ratio(cat_data, num_data)

​                corr[i, j] = r



​    \# 对称

​    for i in range(n_features):

​        for j in range(i):

​            corr[i, j] = corr[j, i]



​    return corr这个里面的类别写的都不对,全变成了feature1-40是怎么回事?



def plot_correlation_heatmap(X, col_names, numeric_cols_idx, filename,

​                             method_numeric="pearson", cmap="ocean",

​                             vmin=-1, vmax=1):

​    """

​    绘制混合变量相关性热力图

​    """

​    ensure_dir_for_file(filename)

​    corr_matrix = mixed_correlation_matrix(

​        X, col_names, numeric_cols_idx,

​        method_numeric=method_numeric

​    )



​    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(col_names)),

​                                   max(8, 0.5 * len(col_names))))



​    sns.heatmap(corr_matrix,

​                xticklabels=col_names,

​                yticklabels=col_names,

​                cmap=cmap,

​                annot=True,

​                fmt=".2f",

​                square=True,

​                vmin=vmin,

​                vmax=vmax,

​                ax=ax,

​                cbar_kws={"shrink": 0.8, "aspect": 30, "label": "Correlation"})



​    ax.set_title("Mixed Correlation Heatmap", fontsize=14)

​    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)

​    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=12)



​    \# 用 subplots_adjust 代替 tight_layout，避免可能的警告

​    fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.12)



​    plt.savefig(filename, dpi=700)

​    plt.close()

​    print(f"[plot_correlation_heatmap] => {filename}")



fn2 = os.path.join(data_corr_dir, "correlation_heatmap_one_hot.jpg")

​        plot_correlation_heatmap(

​            X_onehot,

​            col_names=onehot_colnames,

​            numeric_cols_idx=numeric_idx,

​            filename=fn2

​        )



能看出来出了什么问题吗?



我原来的相关代码:

  \# =========== 2) (可选)相关性图 ===========

​    if config["evaluation"].get("save_correlation", False):

​        df_raw_14 = load_raw_data_for_correlation(csv_path, drop_nan=True)

​        raw_cols = df_raw_14.columns.tolist()

​        numeric_14 = df_raw_14.select_dtypes(include=[np.number]).columns.tolist()

​        numeric_idx_14 = [raw_cols.index(c) for c in numeric_14]



​        fn1 = os.path.join(data_corr_rel, "correlation_heatmap.jpg")

​        plot_correlation_heatmap(df_raw_14.values, col_names=raw_cols,

​                                 numeric_cols_idx=numeric_idx_14, filename=fn1)



​        fn2 = os.path.join(data_corr_rel, "correlation_heatmap_one_hot.jpg")

​        plot_correlation_heatmap(X, col_names=x_col_names,

​                                 numeric_cols_idx=range(X.shape[1]),

​                                 filename=fn2)



你帮我看看之前有什么问题吗?发现以后帮我修改下我的visualization.py吧



请帮我分析下咋回事,此外,我发现我的图片都还算保存的完整,但是我的图片显示不全,有一部分被截断了基本上所有的图片都有类似的问题



请在我的代码上面修改,如果有些内容没有加载成功请帮我修改:

目前我的图片内容只有correlation_heatmap_one_hot.jpg有问题,里面的参数名字没有正确读出,而是一堆feature1..40其余的图片都画得很好,内容什么的也都正确,就是图片有一些colorbar遮挡,有一些显示不全的问题.



我现在给你我的完整代码:



"""

data_preprocessing/data_loader.py



Revised version to explicitly:

1) Only take first 14 columns from CSV (0..13)
2) Drop or fill NaNs in those columns
3) Then parse columns 0..9 => X_df, columns 10..13 => Y
4) Perform one-hot on X_df
5) Return (X, Y, numeric_cols_idx, x_col_names, y_col_names)
6) 额外增加了一个函数 extract_data_statistics, 用于记录每个连续列的 min、max、mean,

   以及把one-hot列按组分开，以便在推理时使用。

"""



import pandas as pd

import numpy as np



def load_dataset(csv_path, drop_nan=True):

​    """

​    Read a CSV file, forcibly keep only the first 14 columns.

​    Then parse the first 10 as X, the next 4 as Y.

​    Perform one-hot encoding for categorical features in X.

​    Finally, return X (NumPy), Y (NumPy), numeric_cols_idx, x_col_names, y_col_names.

​    """



​    \# 1) 读取CSV, 强制只保留前14列

​    df_raw = pd.read_csv(csv_path)

​    df = df_raw.iloc[:, :14].copy()



​    \# 2) 处理空值

​    if drop_nan:

​        df.dropna(subset=df.columns, how='any', inplace=True)



​    \# 3) 拆分 X_df, Y_df

​    if df.shape[1] < 14:

​        raise ValueError("After dropping/cleaning, not enough columns remain (need 14).")



​    X_df = df.iloc[:, :10].copy()

​    Y_df = df.iloc[:, 10:14].copy()  # shape: (N,4)

​    y_col_names = list(Y_df.columns)



​    Y = Y_df.values



​    \# (可选)检查 Y 是否有负值或>100

​    for i, cname in enumerate(y_col_names):

​        below0 = (Y[:,i] < 0).sum()

​        above100 = (Y[:,i] > 100).sum()

​        if below0>0 or above100>0:

​            print(f"[WARN] Column '{cname}' => {below0} vals<0, {above100} vals>100 total.")



​    \# 4) 分析 X_df

​    categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()

​    numeric_cols_original = [col for col in X_df.columns if col not in categorical_cols]



​    \# 5) One-hot

​    X_encoded = pd.get_dummies(X_df, columns=categorical_cols)

​    all_cols = X_encoded.columns.tolist()



​    numeric_cols_idx = []

​    for i, colname in enumerate(all_cols):

​        if colname in numeric_cols_original:

​            numeric_cols_idx.append(i)



​    X = X_encoded.values

​    x_col_names = list(X_encoded.columns)



​    return X, Y, numeric_cols_idx, x_col_names, y_col_names





def load_raw_data_for_correlation(csv_path, drop_nan=True):

​    """

​    如果要做混合变量相关性分析，而不想做 One-Hot，可用这个读取前14列。

​    """

​    df_raw = pd.read_csv(csv_path)

​    df = df_raw.iloc[:, :14].copy()

​    if drop_nan:

​        df.dropna(subset=df.columns, how='any', inplace=True)

​    return df





def extract_data_statistics(X, x_col_names, numeric_cols_idx):

​    """

​    提取并返回一个 dict:

​      {

​        "continuous_cols": {

​            "colname": {

​                "min": ...,

​                "max": ...,

​                "mean": ...

​            }, ...

​        },

​        "onehot_groups": [

​            ...

​        ]

​      }

​    用于推理时做网格/枚举.

​    """

​    stats = {

​        "continuous_cols": {},

​        "onehot_groups": []

​    }



​    \# 1) 连续列信息

​    for idx in numeric_cols_idx:

​        cname = x_col_names[idx]

​        col_data = X[:, idx]

​        stats["continuous_cols"][cname] = {

​            "min": float(np.min(col_data)),

​            "max": float(np.max(col_data)),

​            "mean": float(np.mean(col_data))

​        }



​    \# 2) one-hot分组

​    prefix_map = {}

​    for i, cname in enumerate(x_col_names):

​        if i not in numeric_cols_idx:

​            \# 说明这是 one-hot 列

​            if '_' in cname:

​                prefix = cname.split('_')[0]

​            else:

​                prefix = cname

​            prefix_map.setdefault(prefix, []).append(i)



​    for pref, idxs in prefix_map.items():

​        if len(idxs)>=2:

​            stats["onehot_groups"].append(sorted(idxs))

​        else:

​            stats["onehot_groups"].append([idxs[0]])



​    return stats



"""

data_preprocessing/data_split.py



Contains the function to split data into train/validation sets

using scikit-learn's train_test_split.

Split data into train/val sets or K folds

"""



from sklearn.model_selection import train_test_split, KFold



def split_data(X, Y, test_size=0.2, random_state=42):

​    """

​    Split the dataset into train and validation sets.

​    :param X: Input features, shape (N, input_dim)

​    :param Y: Output targets, shape (N, output_dim)

​    :param test_size: Fraction of data for validation

​    :param random_state: For reproducibility

​    :return: X_train, X_val, Y_train, Y_val

​    """

​    return train_test_split(X, Y, test_size=test_size, random_state=random_state)





"""

data_preprocessing/my_dataset.py



Defines a custom PyTorch Dataset to facilitate DataLoader usage

when training with PyTorch-based models (like the ANN).

"""



import torch

from torch.utils.data import Dataset



class MyDataset(Dataset):

​    """

​    A simple Dataset for multi-output regression using PyTorch.

​    """

​    def __init__(self, X, Y):

​        """

​        Constructor for MyDataset.

​        :param X: NumPy array of input features

​        :param Y: NumPy array of output targets

​        """

​        \# Convert to torch.float32 Tensors

​        self.X = torch.tensor(X, dtype=torch.float32)

​        self.Y = torch.tensor(Y, dtype=torch.float32)



​    def __len__(self):

​        """

​        Return the total number of samples.

​        """

​        return len(self.X)



​    def __getitem__(self, idx):

​        """

​        Return a single sample (x, y) at index idx.

​        """

​        return self.X[idx], self.Y[idx]



"""

data_preprocessing/scaler_utils.py



Contains functions for data standardization (using StandardScaler),

as well as saving/loading scaler objects for future use.



【去掉原先的 logit transform】, 保留/新增 bounded transform(0..100 => -1..1).

"""



import numpy as np

from sklearn.preprocessing import StandardScaler

import joblib



def bounded_transform(y):

​    """

​    y in [0,100] => z in [-1,1], linear:

​        z = 2*(y/100) - 1

​    clamp y to [0,100] just in case

​    """

​    y_ = np.clip(y, 0, 100)

​    return 2*(y_/100.0) - 1



def inverse_bounded_transform(z):

​    """

​    z => clamp in [-1,1], => y=100*(z+1)/2 in [0,100]

​    """

​    z_ = np.clip(z, -1, 1)

​    return 100.0*(z_+1.0)/2.0



def standardize_data(X_train, X_val,

​                     Y_train, Y_val,

​                     do_input=True,

​                     do_output=False,

​                     numeric_cols_idx=None,

​                     do_output_bounded=False):

​    """

​    Optionally standardize input features (X) and/or output targets (Y).

​    If do_output_bounded=True => 0..100 => -1..1 => standard => model.

​    """

​    scaler_x = None

​    scaler_y = None



​    X_train_s = np.copy(X_train)

​    X_val_s   = np.copy(X_val)

​    Y_train_s = np.copy(Y_train)

​    Y_val_s   = np.copy(Y_val)



​    if do_input:

​        if numeric_cols_idx is None:

​            numeric_cols_idx = list(range(X_train.shape[1]))

​        scaler_x = StandardScaler()

​        scaler_x.fit(X_train_s[:, numeric_cols_idx])

​        X_train_s[:, numeric_cols_idx] = scaler_x.transform(X_train_s[:, numeric_cols_idx])

​        X_val_s[:, numeric_cols_idx]   = scaler_x.transform(X_val_s[:, numeric_cols_idx])



​    if do_output:

​        \# bounded or not

​        if do_output_bounded:

​            \# 0..100 => -1..1

​            for i in range(Y_train_s.shape[1]):

​                Y_train_s[:,i] = bounded_transform(Y_train_s[:,i])

​                Y_val_s[:,i]   = bounded_transform(Y_val_s[:,i])

​            transform_type = "bounded+standard"

​        else:

​            transform_type = "standard"



​        \# standard

​        scaler_obj = StandardScaler()

​        scaler_obj.fit(Y_train_s)

​        Y_train_s = scaler_obj.transform(Y_train_s)

​        Y_val_s   = scaler_obj.transform(Y_val_s)



​        scaler_y = {

​            "type": transform_type,

​            "scaler": scaler_obj

​        }



​    return (X_train_s, X_val_s, scaler_x), (Y_train_s, Y_val_s, scaler_y)



def save_scaler(scaler, path):

​    if scaler is not None:

​        joblib.dump(scaler, path)



def load_scaler(path):

​    return joblib.load(path)



def inverse_transform_output(y_pred, scaler_y):

​    """

​    If scaler_y["type"]=="bounded+standard": inverse standard => inverse_bounded => [0,100].

​    If "standard": just inverse standard.

​    """

​    if scaler_y is None:

​        return y_pred

​    if not isinstance(scaler_y, dict):

​        \# older usage => direct standard

​        return scaler_y.inverse_transform(y_pred)



​    transform_type = scaler_y["type"]

​    scaler_obj = scaler_y["scaler"]

​    \# 1) inverse standard

​    y_ = scaler_obj.inverse_transform(y_pred)



​    if transform_type.startswith("bounded"):

​        \# each col => clamp => [0,100]

​        for i in range(y_.shape[1]):

​            y_[:,i] = inverse_bounded_transform(y_[:,i])



​    return y_





"""

evaluation/metrics.py



Implements common regression metrics: MSE, MAE, R2.

"""



import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def compute_regression_metrics(y_true, y_pred):

​    """

​    Compute MSE, MAE, and R2 for the given true/predicted values.

​    :param y_true: shape (N, output_dim)

​    :param y_pred: shape (N, output_dim)

​    :return: a dict with "MSE", "MAE", "R2"

​    """

​    mse = mean_squared_error(y_true, y_pred)

​    mae = mean_absolute_error(y_true, y_pred)

​    r2  = r2_score(y_true, y_pred, multioutput='uniform_average')

​    return {

​        "MSE": mse,

​        "MAE": mae,

​        "R2" : r2

​    }





if __name__ == "__main__":

​    \# 假设真实值和预测值

​    y_true = np.array([[3.5, 2.1], [4.0, 3.3], [5.2, 6.8]])

​    y_pred = np.array([[3.7, 2.0], [4.1, 3.5], [5.0, 6.5]])



​    \# 计算回归指标

​    metrics = compute_regression_metrics(y_true, y_pred)

​    print(metrics)





"""

torch_losses.py



Implements PyTorch-specific loss functions (MSE, MAE, etc.).

"""



import torch.nn.functional as F



def mse_loss(pred, target):

​    """

​    Mean Squared Error.

​    """

​    return F.mse_loss(pred, target)



def mae_loss(pred, target):

​    """

​    Mean Absolute Error.

​    """

​    return F.l1_loss(pred, target)



def get_torch_loss_fn(name="mse"):

​    """

​    Return the specified PyTorch loss function by name.

​    :param name: "mse" or "mae"

​    :return: callable that takes (pred, target) -> scalar loss

​    """

​    if name.lower() == "mse":

​        return mse_loss

​    elif name.lower() == "mae":

​        return mae_loss

​    else:

​        raise ValueError(f"Unknown loss: {name}")





"""

models/model_ann.py



Implementation of an ANN for multi-output regression with optional Dropout.

"""

import torch

import torch.nn as nn



class ANNRegression(nn.Module):

​    def __init__(self,

​                 input_dim,

​                 output_dim,

​                 hidden_dims=[32,64,32],

​                 dropout=0.0,

​                 activation="ReLU",

​                 random_seed=None):

​        """

​        :param input_dim:  输入特征维度

​        :param output_dim: 输出维度

​        :param hidden_dims: 隐藏层大小列表

​        :param dropout: Dropout 概率, e.g. 0.2

​        :param activation: 激活函数名称, e.g. "ReLU" / "Tanh" / "Sigmoid"

​        :param random_seed: 若指定, 则使用该种子来初始化网络权重

​        """

​        super().__init__()



​        if random_seed is not None:

​            torch.manual_seed(random_seed)



​        act_fn = None

​        if activation.lower() == "relu":

​            act_fn = nn.ReLU

​        elif activation.lower() == "tanh":

​            act_fn = nn.Tanh

​        elif activation.lower() == "sigmoid":

​            act_fn = nn.Sigmoid

​        else:

​            act_fn = nn.ReLU



​        layers = []

​        prev_dim = input_dim

​        for hd in hidden_dims:

​            layers.append(nn.Linear(prev_dim, hd))

​            layers.append(act_fn())

​            if dropout > 0.0:

​                layers.append(nn.Dropout(dropout))

​            prev_dim = hd

​        layers.append(nn.Linear(prev_dim, output_dim))



​        self.net = nn.Sequential(*layers)



​    def forward(self, x):

​        return self.net(x)





"""

models/model_catboost.py

"""



from catboost import CatBoostRegressor



class CatBoostRegression:

​    """

​    CatBoost with l2_leaf_reg controlling L2 regularization strength.

​    """

​    def __init__(self, iterations=100, learning_rate=0.1, depth=6,

​                 random_seed=42, l2_leaf_reg=3.0):

​        self.model = CatBoostRegressor(

​            iterations=iterations,

​            learning_rate=learning_rate,

​            depth=depth,

​            random_seed=random_seed,

​            verbose=0,

​            loss_function="MultiRMSE",

​            l2_leaf_reg=l2_leaf_reg

​        )



​    def fit(self, X, Y):

​        self.model.fit(X, Y)



​    def predict(self, X):

​        return self.model.predict(X)



​    \# @property

​    \# def feature_importances_(self):

​    \#     return self.model.get_feature_importance(type="PredictionValuesChange")



​    @property

​    def feature_importances_(self):

​        """

​        Returns normalized feature importance values for CatBoost, ensuring

​        the sum of all importances equals 1, making it comparable to

​        feature importances from XGBoost, RandomForest, and DecisionTree.

​        """

​        importances = self.model.get_feature_importance(type="PredictionValuesChange")



​        \# 避免除零错误

​        total_importance = sum(importances)

​        if total_importance > 0:

​            return importances / total_importance  # 归一化，使总和为 1

​        else:

​            return importances  # 如果全是 0，就直接返回







from sklearn.tree import DecisionTreeRegressor



class DTRegression:

​    """

​    Decision Tree with optional ccp_alpha (cost-complexity pruning).

​    """

​    def __init__(self, max_depth=None, random_state=42, ccp_alpha=0.0):

​        self.model = DecisionTreeRegressor(

​            max_depth=max_depth,

​            random_state=random_state,

​            ccp_alpha=ccp_alpha

​        )



​    def fit(self, X, Y):

​        self.model.fit(X, Y)



​    def predict(self, X):

​        return self.model.predict(X)



​    @property

​    def feature_importances_(self):

​        return self.model.feature_importances_





"""

models/model_rf.py



A Random Forest regressor wrapper using scikit-learn.

"""



from sklearn.ensemble import RandomForestRegressor



class RFRegression:

​    """

​    A simple regression model using RandomForest from scikit-learn.

​    """



​    def __init__(self, n_estimators=100, max_depth=10, random_state=42,

​                 ccp_alpha=0.0, min_samples_leaf=1):

​        """

​               :param n_estimators: number of trees in the forest

​               :param max_depth: maximum depth of the trees

​               :param random_state: seed for reproducibility

​               """

​        self.model = RandomForestRegressor(

​            n_estimators=n_estimators,

​            max_depth=max_depth,

​            random_state=random_state,

​            ccp_alpha=ccp_alpha,

​            min_samples_leaf=min_samples_leaf

​        )



​    def fit(self, X, Y):

​        """

​        Train the Random Forest model.

​        :param X: training inputs, shape (N, input_dim)

​        :param Y: training targets, shape (N, output_dim)

​        """

​        self.model.fit(X, Y)



​    def predict(self, X):

​        """

​        Predict using the trained Random Forest model.

​        :param X: inputs, shape (N, input_dim)

​        :return: predictions, shape (N, output_dim)

​        """

​        return self.model.predict(X)



​    @property

​    def feature_importances_(self):

​        """

​        方便外部直接访问 => rf_model.feature_importances_

​        """

​        return self.model.feature_importances_



"""

inference.py



需求:

1) 读取 config.yaml
2) 读取 ./models/<model_type>/trained_model.pkl (或 best_ann.pt) + scaler + colnames + metadata.pkl
3) 分别计算:

   (a) 2D Heatmap => heatmap_pred.npy, grid_x.npy, grid_y.npy

   (b) Confusion => confusion_pred.npy

4) 保存到 postprocessing/<csv_name>/inference/<model_type>/
5) 不做任何绘图



已去掉所有 K-Fold 相关逻辑.

"""



import yaml

import os

import numpy as np

import torch

import joblib

from itertools import product

from tqdm import trange



from data_preprocessing.scaler_utils import load_scaler, inverse_transform_output



\# 模型

from models.model_ann import ANNRegression

from models.model_rf import RFRegression

from models.model_dt import DTRegression

from models.model_catboost import CatBoostRegression

from models.model_xgb import XGBRegression



def ensure_dir(path):

​    os.makedirs(path, exist_ok=True)



def load_inference_model(model_type, config):

​    model_dir= os.path.join("./models", model_type)

​    if not os.path.isdir(model_dir):

​        raise FileNotFoundError(f"[ERROR] Directory not found => {model_dir}")



​    x_col_path= os.path.join(model_dir, "x_col_names.npy")

​    y_col_path= os.path.join(model_dir, "y_col_names.npy")

​    if not (os.path.exists(x_col_path) and os.path.exists(y_col_path)):

​        raise FileNotFoundError("[ERROR] x_col_names.npy or y_col_names.npy not found.")



​    x_col_names= list(np.load(x_col_path, allow_pickle=True))

​    y_col_names= list(np.load(y_col_path, allow_pickle=True))



​    if model_type=="ANN":

​        ann_cfg= config["model"]["ann_params"]

​        net= ANNRegression(

​            input_dim=len(x_col_names),

​            output_dim=len(y_col_names),

​            hidden_dims=ann_cfg["hidden_dims"],

​            dropout=ann_cfg.get("dropout",0.0),

​            activation=ann_cfg.get("activation","ReLU"),

​            random_seed=ann_cfg.get("random_seed",42)

​        )

​        ckpt_path= os.path.join(model_dir, "best_ann.pt")

​        if not os.path.exists(ckpt_path):

​            raise FileNotFoundError(f"[ERROR] {ckpt_path} not found.")

​        try:

​            state_dict= torch.load(ckpt_path, map_location="cpu", weights_only=True)

​        except TypeError:

​            state_dict= torch.load(ckpt_path, map_location="cpu")

​        net.load_state_dict(state_dict)

​        net.eval()

​        return net, x_col_names, y_col_names

​    else:

​        pkl_path= os.path.join(model_dir, "trained_model.pkl")

​        if not os.path.exists(pkl_path):

​            raise FileNotFoundError(f"[ERROR] {pkl_path} not found.")

​        model= joblib.load(pkl_path)

​        return model, x_col_names, y_col_names



def model_predict(model, X_2d):

​    if hasattr(model, 'eval') and hasattr(model, 'forward'):

​        with torch.no_grad():

​            t_= torch.tensor(X_2d, dtype=torch.float32)

​            out_= model(t_)

​        return out_.cpu().numpy()

​    else:

​        return model.predict(X_2d)



def inference_main():

​    with open("./configs/config.yaml","r") as f:

​        config= yaml.safe_load(f)



​    inf_models= config["inference"].get("models", [])

​    if not inf_models:

​        print("[INFO] No inference models => exit.")

​        return



​    meta_path= os.path.join("./models","metadata.pkl")

​    if not os.path.exists(meta_path):

​        print(f"[ERROR] metadata => {meta_path} missing.")

​        return

​    stats_dict= joblib.load(meta_path)



​    csv_path= config["data"]["path"]

​    csv_name= os.path.splitext(os.path.basename(csv_path))[0]

​    base_inf= os.path.join("postprocessing", csv_name, "inference")

​    ensure_dir(base_inf)



​    for mtype in inf_models:

​        print(f"\n========== Inference => {mtype} ==========")

​        outdir_m= os.path.join(base_inf, mtype)

​        ensure_dir(outdir_m)



​        try:

​            model, x_col_names, y_col_names= load_inference_model(mtype, config)

​        except FileNotFoundError as e:

​            print(e)

​            continue



​        \# 加载 scaler

​        model_dir= os.path.join("./models", mtype)

​        sx_path= os.path.join(model_dir, f"scaler_x_{mtype}.pkl")

​        sy_path= os.path.join(model_dir, f"scaler_y_{mtype}.pkl")

​        if os.path.exists(sx_path):

​            scaler_x= load_scaler(sx_path)

​        else:

​            scaler_x= None

​        if os.path.exists(sy_path):

​            scaler_y= load_scaler(sy_path)

​        else:

​            scaler_y= None



​        \# numeric cols

​        numeric_cols_idx= []

​        for c_ in stats_dict["continuous_cols"].keys():

​            numeric_cols_idx.append( x_col_names.index(c_) )



​        \# (a) 2D Heatmap

​        conti_x= config["inference"].get("conti_x", "Potential (V vs. RHE)")

​        conti_y= config["inference"].get("conti_y", "Particle size (nm)")

​        n_points= config["inference"].get("n_points", 20)



​        if (conti_x not in stats_dict["continuous_cols"]) or (conti_y not in stats_dict["continuous_cols"]):

​            print(f"[WARN] conti_x={conti_x} or conti_y={conti_y} not found in metadata => skip heatmap.")

​        else:

​            cinfo_x= stats_dict["continuous_cols"][conti_x]

​            cinfo_y= stats_dict["continuous_cols"][conti_y]



​            x_vals= np.linspace(cinfo_x["min"], cinfo_x["max"], n_points)

​            y_vals= np.linspace(cinfo_y["min"], cinfo_y["max"], n_points)

​            grid_x, grid_y= np.meshgrid(x_vals, y_vals)



​            base_vec= np.zeros(len(x_col_names), dtype=float)

​            \# 其它连续 => mean

​            for cname,cstat in stats_dict["continuous_cols"].items():

​                idx_= x_col_names.index(cname)

​                base_vec[idx_]= cstat["mean"]



​            oh_groups= stats_dict["onehot_groups"]

​            all_combos= list(product(*oh_groups))

​            if len(all_combos)>20000:

​                print(f"[WARN] combos too large => {len(all_combos)}, truncated => 20000")

​                all_combos= all_combos[:20000]



​            \# outdim

​            tmp_inp= base_vec.copy().reshape(1,-1)

​            if scaler_x:

​                tmp_inp[:, numeric_cols_idx]= scaler_x.transform(tmp_inp[:, numeric_cols_idx])

​            tmp_out= model_predict(model, tmp_inp)

​            if tmp_out.ndim==1:

​                outdim= tmp_out.shape[0]

​            else:

​                outdim= tmp_out.shape[1]



​            H,W= grid_x.shape

​            heatmap_pred= np.zeros((H,W,outdim), dtype=float)



​            row_iter= trange(H, desc="2DHeatmap Rows", ncols=100)

​            for i in row_iter:

​                for j in range(W):

​                    vec= base_vec.copy()

​                    vec[x_col_names.index(conti_x)] = grid_x[i,j]

​                    vec[x_col_names.index(conti_y)] = grid_y[i,j]



​                    sum_real= np.zeros(outdim, dtype=float)

​                    for combo in all_combos:

​                        tmpv= vec.copy()

​                        \# zero out oh

​                        for grp_ in oh_groups:

​                            for cid_ in grp_:

​                                tmpv[cid_]=0

​                        \# activate

​                        for colid_ in combo:

​                            tmpv[colid_]=1



​                        inp_2d= tmpv.reshape(1,-1)

​                        if scaler_x:

​                            inp_2d[:, numeric_cols_idx]= scaler_x.transform(inp_2d[:, numeric_cols_idx])

​                        scaled_pred= model_predict(model, inp_2d)

​                        real_pred= inverse_transform_output(scaled_pred, scaler_y)

​                        if real_pred.ndim==2:

​                            sum_real+= real_pred[0]

​                        else:

​                            sum_real+= real_pred



​                    avg_real= sum_real/ len(all_combos)

​                    avg_real= np.clip(avg_real,0,100)

​                    heatmap_pred[i,j,:]= avg_real



​            np.save(os.path.join(outdir_m,"heatmap_pred.npy"), heatmap_pred)

​            np.save(os.path.join(outdir_m,"grid_x.npy"), grid_x)

​            np.save(os.path.join(outdir_m,"grid_y.npy"), grid_y)

​            print(f"[INFO] 2D heatmap => shape={heatmap_pred.shape} saved => {outdir_m}")



​        \# (b) Confusion => group1_index=0, group2_index=1

​        if len(stats_dict["onehot_groups"])>=2:

​            grpA= stats_dict["onehot_groups"][0]

​            grpB= stats_dict["onehot_groups"][1]

​            rest= []

​            for i,gg in enumerate(stats_dict["onehot_groups"]):

​                if i not in [0,1]:

​                    rest.append(gg)



​            rest_combos= list(product(*rest))

​            if len(rest_combos)>20000:

​                rest_combos= rest_combos[:20000]



​            \# outdim

​            tmp_inp= base_vec.copy().reshape(1,-1)

​            if scaler_x:

​                tmp_inp[:, numeric_cols_idx]= scaler_x.transform(tmp_inp[:, numeric_cols_idx])

​            tmp_out= model_predict(model, tmp_inp)

​            if tmp_out.ndim==1:

​                outdim= tmp_out.shape[0]

​            else:

​                outdim= tmp_out.shape[1]



​            n_rows= len(grpA)

​            n_cols= len(grpB)

​            confusion_pred= np.zeros((n_rows,n_cols,outdim), dtype=float)



​            row_iter= trange(n_rows, desc="Confusion Rows", ncols=100)

​            for i in row_iter:

​                rcid= grpA[i]

​                for j in range(n_cols):

​                    ccid= grpB[j]

​                    sum_real= np.zeros(outdim, dtype=float)

​                    for combo in rest_combos:

​                        tmpv= base_vec.copy()

​                        \# zero out

​                        for g_ in oh_groups:

​                            for cid_ in g_:

​                                tmpv[cid_]=0

​                        tmpv[rcid]=1

​                        tmpv[ccid]=1

​                        for colid_ in combo:

​                            tmpv[colid_]=1



​                        inp_2d= tmpv.reshape(1,-1)

​                        if scaler_x:

​                            inp_2d[:, numeric_cols_idx]= scaler_x.transform(inp_2d[:, numeric_cols_idx])

​                        scaled_out= model_predict(model, inp_2d)

​                        real_out= inverse_transform_output(scaled_out, scaler_y)

​                        if real_out.ndim==2:

​                            sum_real+= real_out[0]

​                        else:

​                            sum_real+= real_out



​                    avg_real= sum_real/len(rest_combos)

​                    avg_real= np.clip(avg_real,0,100)

​                    confusion_pred[i,j,:]= avg_real



​            np.save(os.path.join(outdir_m,"confusion_pred.npy"), confusion_pred)

​            print(f"[INFO] confusion => shape={confusion_pred.shape}, saved => {outdir_m}")

​        else:

​            print("[WARN] Not enough onehot groups => skip confusion.")





if __name__=="__main__":

​    inference_main()





"""

train.py



需求:

1) 在 config.yaml 的 data.path 中读取 CSV 数据.
2) 仅执行模型训练 + 将训练/验证阶段的预测、指标、损失等保存到:

   postprocessing/<csv_name>/train/<model_type>/

3) 同时保存:

   \- df_raw_14.csv => 原始前14列(做可视化时使用)

   \- X_onehot.npy => 训练时用到的一份 OneHot X（供可视化时做 correlation_heatmap_one_hot）

   \- Y_train.npy, Y_val.npy => 用于可视化散点/残差等

4) 不直接绘图, 只保存文件.



已去掉所有 K-Fold 相关逻辑.

"""



import yaml

import os

import numpy as np

import torch

import joblib



from data_preprocessing.data_loader import (

​    load_dataset,

​    load_raw_data_for_correlation,

​    extract_data_statistics

)

from data_preprocessing.data_split import split_data

from data_preprocessing.my_dataset import MyDataset

from data_preprocessing.scaler_utils import (

​    standardize_data, inverse_transform_output, save_scaler

)



\# 模型

from models.model_ann import ANNRegression

from models.model_rf import RFRegression

from models.model_dt import DTRegression

from models.model_catboost import CatBoostRegression

from models.model_xgb import XGBRegression



\# 损失 & 训练

from losses.torch_losses import get_torch_loss_fn

from trainers.train_torch import train_torch_model_dataloader

from trainers.train_sklearn import train_sklearn_model



\# 指标

from evaluation.metrics import compute_regression_metrics



def ensure_dir(path):

​    os.makedirs(path, exist_ok=True)



def create_model_by_type(model_type, config, random_seed=42, input_dim=None):

​    """

​    根据model_type创建对应模型实例.

​    去掉K-Fold后不影响此函数。

​    """

​    if model_type == "ANN":

​        ann_cfg = config["model"]["ann_params"]

​        actual_dim = input_dim if input_dim is not None else ann_cfg["input_dim"]

​        model = ANNRegression(

​            input_dim=actual_dim,

​            output_dim=ann_cfg["output_dim"],

​            hidden_dims=ann_cfg["hidden_dims"],

​            dropout=ann_cfg.get("dropout", 0.0),

​            activation=ann_cfg.get("activation", "ReLU"),

​            random_seed=ann_cfg.get("random_seed", 42)

​        )

​        return model

​    elif model_type == "RF":

​        rf_cfg = config["model"]["rf_params"]

​        return RFRegression(

​            n_estimators=rf_cfg["n_estimators"],

​            max_depth=rf_cfg["max_depth"],

​            random_state=rf_cfg["random_state"],

​            ccp_alpha=rf_cfg.get("ccp_alpha", 0.0),

​            min_samples_leaf=rf_cfg.get("min_samples_leaf", 1)

​        )

​    elif model_type == "DT":

​        dt_cfg = config["model"]["dt_params"]

​        return DTRegression(

​            max_depth=dt_cfg["max_depth"],

​            random_state=dt_cfg["random_state"],

​            ccp_alpha=dt_cfg.get("ccp_alpha", 0.0)

​        )

​    elif model_type == "CatBoost":

​        cat_cfg = config["model"]["catboost_params"]

​        return CatBoostRegression(

​            iterations=cat_cfg["iterations"],

​            learning_rate=cat_cfg["learning_rate"],

​            depth=cat_cfg["depth"],

​            random_seed=cat_cfg["random_seed"],

​            l2_leaf_reg=cat_cfg.get("l2_leaf_reg", 3.0)

​        )

​    elif model_type == "XGB":

​        xgb_cfg = config["model"]["xgb_params"]

​        return XGBRegression(

​            n_estimators=xgb_cfg["n_estimators"],

​            learning_rate=xgb_cfg["learning_rate"],

​            max_depth=xgb_cfg["max_depth"],

​            random_state=xgb_cfg["random_seed"],

​            reg_alpha=xgb_cfg.get("reg_alpha", 0.0),

​            reg_lambda=xgb_cfg.get("reg_lambda", 1.0)

​        )

​    else:

​        raise ValueError(f"Unknown model type: {model_type}")



def train_main():

​    with open("./configs/config.yaml","r") as f:

​        config = yaml.safe_load(f)



​    csv_path = config["data"]["path"]

​    csv_name = os.path.splitext(os.path.basename(csv_path))[0]

​    base_outdir = os.path.join("postprocessing", csv_name, "train")

​    ensure_dir(base_outdir)



​    \# 1) 加载数据 (OneHot)

​    X, Y, numeric_cols_idx, x_col_names, y_col_names = load_dataset(csv_path)

​    \#   1.1) 保存一份 X_onehot.npy => 做 correlation_heatmap_one_hot 用

​    np.save(os.path.join(base_outdir, "X_onehot.npy"), X)



​    \# 1.2) 如果要做 raw correlation, 可以额外保存 df_raw_14.csv => 供可视化

​    df_raw_14 = load_raw_data_for_correlation(csv_path, drop_nan=True)

​    raw_csv_path = os.path.join(base_outdir, "df_raw_14.csv")

​    df_raw_14.to_csv(raw_csv_path, index=False)

​    print(f"[INFO] Saved raw 14-col CSV => {raw_csv_path}")



​    \# 2) 提取统计信息 => metadata.pkl (供 inference 使用)

​    stats_dict = extract_data_statistics(X, x_col_names, numeric_cols_idx)

​    meta_path = os.path.join("./models","metadata.pkl")

​    joblib.dump(stats_dict, meta_path)

​    print(f"[INFO] metadata saved => {meta_path}")



​    \# 3) 数据拆分 & 标准化

​    random_seed = config["data"].get("random_seed", 42)

​    X_train, X_val, Y_train, Y_val = split_data(

​        X, Y,

​        test_size=config["data"]["test_size"],

​        random_state=random_seed

​    )

​    (X_train_s, X_val_s, sx), (Y_train_s, Y_val_s, sy) = standardize_data(

​        X_train, X_val, Y_train, Y_val,

​        do_input=config["preprocessing"]["standardize_input"],

​        do_output=config["preprocessing"]["standardize_output"],

​        numeric_cols_idx=numeric_cols_idx,

​        do_output_bounded=config["preprocessing"].get("bounded_output", False)

​    )



​    \# 3.1) 保存 Y_train, Y_val => 供可视化(散点/残差)

​    np.save(os.path.join(base_outdir, "Y_train.npy"), Y_train)

​    np.save(os.path.join(base_outdir, "Y_val.npy"),   Y_val)



​    \# 4) 训练 & 保存

​    model_types = config["model"]["types"]

​    for mtype in model_types:

​        print(f"\n=== Train model: {mtype} ===")

​        outdir_m = os.path.join(base_outdir, mtype)

​        ensure_dir(outdir_m)



​        model = create_model_by_type(mtype, config, random_seed, input_dim=X_train_s.shape[1])



​        if mtype=="ANN":

​            from losses.torch_losses import get_torch_loss_fn

​            loss_fn= get_torch_loss_fn(config["loss"]["type"])

​            ann_cfg= config["model"]["ann_params"]

​            from data_preprocessing.my_dataset import MyDataset

​            train_ds= MyDataset(X_train_s, Y_train_s)

​            val_ds=   MyDataset(X_val_s,   Y_val_s)



​            model, train_losses, val_losses = train_torch_model_dataloader(

​                model, train_ds, val_ds,

​                loss_fn=loss_fn,

​                epochs=ann_cfg["epochs"],

​                batch_size=ann_cfg["batch_size"],

​                lr=float(ann_cfg["learning_rate"]),

​                weight_decay=float(ann_cfg.get("weight_decay",0.0)),

​                checkpoint_path=ann_cfg["checkpoint_path"],

​                log_interval=config["training"]["log_interval"],

​                early_stopping=ann_cfg.get("early_stopping", False),

​                patience=ann_cfg.get("patience",5),

​                optimizer_name=ann_cfg.get("optimizer","Adam")

​            )

​            model.to("cpu")



​            \# 保存 losses

​            np.save(os.path.join(outdir_m, "train_losses.npy"), train_losses)

​            np.save(os.path.join(outdir_m, "val_losses.npy"),   val_losses)

​        else:

​            model = train_sklearn_model(model, X_train_s, Y_train_s)



​        \# 推断(Train/Val)

​        if hasattr(model, 'eval') and hasattr(model, 'forward'):

​            with torch.no_grad():

​                p_tr = model(torch.tensor(X_train_s, dtype=torch.float32)).cpu().numpy()

​                p_va = model(torch.tensor(X_val_s,   dtype=torch.float32)).cpu().numpy()

​            train_pred = p_tr

​            val_pred   = p_va

​        else:

​            train_pred = model.predict(X_train_s)

​            val_pred   = model.predict(X_val_s)



​        \# 反变换

​        if config["preprocessing"]["standardize_output"]:

​            train_pred= inverse_transform_output(train_pred, sy)

​            val_pred=   inverse_transform_output(val_pred, sy)



​        \# 计算metrics

​        train_m= compute_regression_metrics(Y_train, train_pred)

​        val_m=   compute_regression_metrics(Y_val,   val_pred)

​        print(f"   => train={train_m}, val={val_m}")



​        \# 保存预测结果

​        np.save(os.path.join(outdir_m, "train_pred.npy"), train_pred)

​        np.save(os.path.join(outdir_m, "val_pred.npy"),   val_pred)



​        \# 保存 metrics

​        joblib.dump({"train_metrics":train_m, "val_metrics":val_m},

​                    os.path.join(outdir_m, "metrics.pkl"))



​        \# 保存模型 & scaler

​        model_dir= os.path.join("./models", mtype)

​        os.makedirs(model_dir, exist_ok=True)

​        save_scaler(sx, os.path.join(model_dir, f"scaler_x_{mtype}.pkl"))

​        save_scaler(sy, os.path.join(model_dir, f"scaler_y_{mtype}.pkl"))

​        np.save(os.path.join(model_dir, "x_col_names.npy"), x_col_names)

​        np.save(os.path.join(model_dir, "y_col_names.npy"), y_col_names)



​        if mtype!="ANN":

​            joblib.dump(model, os.path.join(model_dir, "trained_model.pkl"))

​            print(f"[INFO] saved => {mtype}/trained_model.pkl")



​    print("\n[INFO] train_main => done. All results saved in postprocessing/<csv_name>/train/ ...")





if __name__=="__main__":

​    train_main()





\# utils.py



"""

utils.py



包含所有绘图函数 & 一些辅助:

1) correlation_heatmap (含普通 & onehot)
2) 训练可视化: loss_curve, scatter(MAE/MSE), residual, feature_importance, etc.
3) 原始数据分析(kde, scatter, boxplot)
4) 推理可视化(2D Heatmap + ConfusionMatrix)
5) 混淆矩阵中在每个三角形内显示数值 + colorbar范围扩展 + 保持正方形布局.



已去掉K-Fold, 保留注释.

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





def ensure_dir(path):

​    os.makedirs(path, exist_ok=True)



def ensure_dir_for_file(filepath):

​    dir_ = os.path.dirname(filepath)

​    if dir_:

​        os.makedirs(dir_, exist_ok=True)



\# --------------------- correlation ---------------------

def cramers_v(x, y):

​    """

​    类别-类别相关性 (Cramer's V)

​    """

​    confusion_matrix = pd.crosstab(x, y)

​    chi2 = ss.chi2_contingency(confusion_matrix)[0]

​    n = confusion_matrix.sum().sum()

​    r, k = confusion_matrix.shape

​    phi2 = max(0, chi2 - (k - 1)*(r - 1)/(n-1))

​    r_adj = r - (r-1)**2/(n-1)

​    k_adj = k - (k-1)**2/(n-1)

​    denom = min(k_adj-1, r_adj-1)

​    if denom == 0:

​        return 0.0

​    else:

​        return math.sqrt(phi2/denom)



def correlation_ratio(cat_data, num_data):

​    """

​    类别-数值相关性 (Correlation Ratio)

​    """

​    df = pd.DataFrame({'cat': cat_data, 'val': num_data})

​    df.dropna(inplace=True)

​    group_means = df.groupby('cat')['val'].mean()

​    mean_all = df['val'].mean()



​    ss_between = 0

​    for cat_value, group_mean in group_means.items():

​        group = df[df['cat'] == cat_value]

​        n = len(group)

​        ss_between += n * (group_mean - mean_all) ** 2



​    ss_total = ((df['val'] - mean_all) ** 2).sum()

​    if ss_total == 0:

​        return 0.0

​    return math.sqrt(ss_between / ss_total)



def mixed_correlation_matrix(X, col_names, numeric_cols_idx,

​                             method_numeric="pearson",

​                             dropna=True):

​    """

​    计算混合变量相关性矩阵: 数值-数值 => Pearson/Spearman,

​                       类别-类别 => Cramer's V,

​                       数值-类别 => Correlation Ratio.

​    """

​    n_features = X.shape[1]

​    if not isinstance(X, pd.DataFrame):

​        X = pd.DataFrame(X, columns=col_names)



​    numeric_set = set(numeric_cols_idx)

​    corr = np.zeros((n_features, n_features), dtype=float)



​    for i in range(n_features):

​        for j in range(n_features):

​            if i == j:

​                corr[i, j] = 1.0

​                continue

​            if i > j:

​                corr[i, j] = corr[j, i]

​                continue



​            col_i = col_names[i]

​            col_j = col_names[j]

​            data_i = X[col_i]

​            data_j = X[col_j]

​            if dropna:

​                valid = ~data_i.isna() & ~data_j.isna()

​                data_i = data_i[valid]

​                data_j = data_j[valid]



​            i_is_num = (i in numeric_set)

​            j_is_num = (j in numeric_set)



​            if i_is_num and j_is_num:

​                \# 数值-数值

​                if method_numeric.lower() == "pearson":

​                    r, _ = ss.pearsonr(data_i, data_j)

​                else:

​                    r, _ = ss.spearmanr(data_i, data_j)

​                corr[i, j] = r

​            elif (not i_is_num) and (not j_is_num):

​                \# 类别-类别

​                r = cramers_v(data_i, data_j)

​                corr[i, j] = r

​            else:

​                \# 数值-类别 => correlation_ratio

​                if i_is_num:

​                    cat_data, num_data = data_j, data_i

​                else:

​                    cat_data, num_data = data_i, data_j

​                r = correlation_ratio(cat_data, num_data)

​                corr[i, j] = r



​    \# 对称

​    for i in range(n_features):

​        for j in range(i):

​            corr[i, j] = corr[j, i]



​    return corr



def plot_correlation_heatmap(X, col_names, numeric_cols_idx, filename,

​                             method_numeric="pearson", cmap="ocean",

​                             vmin=-1, vmax=1):

​    """

​    绘制混合变量相关性热力图

​    """

​    ensure_dir_for_file(filename)

​    corr_matrix = mixed_correlation_matrix(

​        X, col_names, numeric_cols_idx,

​        method_numeric=method_numeric

​    )



​    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(col_names)),

​                                   max(8, 0.5 * len(col_names))))



​    sns.heatmap(corr_matrix,

​                xticklabels=col_names,

​                yticklabels=col_names,

​                cmap=cmap,

​                annot=True,

​                fmt=".2f",

​                square=True,

​                vmin=vmin,

​                vmax=vmax,

​                ax=ax,

​                cbar_kws={"shrink": 0.8, "aspect": 30, "label": "Correlation"})



​    ax.set_title("Mixed Correlation Heatmap", fontsize=14)

​    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)

​    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=12)



​    \# 用 subplots_adjust 代替 tight_layout，避免可能的警告

​    fig.subplots_adjust(left=0.12, right=0.95, top=0.93, bottom=0.12)



​    plt.savefig(filename, dpi=700)

​    plt.close()

​    print(f"[plot_correlation_heatmap] => {filename}")





\# ------------------- 训练可视化: Loss, scatter, residual, etc. -------------------

def plot_loss_curve(train_losses, val_losses, filename):

​    ensure_dir_for_file(filename)

​    fig, ax = plt.subplots(figsize=(6,4))

​    ax.plot(train_losses, label='Train Loss', linewidth=2)

​    ax.plot(val_losses, label='Val Loss', linewidth=2)

​    ax.set_xlabel("Epoch")

​    ax.set_ylabel("Loss")

​    ax.legend()

​    ax.set_title("Training/Validation Loss")



​    fig.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.15)

​    plt.savefig(filename, dpi=700, format='jpg')

​    plt.close()



def plot_scatter_3d_outputs_mse(y_true, y_pred, y_labels=None,

​                                filename="scatter_3d_mse.jpg"):

​    """

​    对多输出回归, 每个输出画一个子图:

​    \- X轴: True

​    \- Y轴: Pred

​    \- 颜色: (y_true - y_pred)^2

​    \- 标题显示 R²

​    \- 可传入 y_labels 显示真实列名

​    """

​    ensure_dir_for_file(filename)

​    if y_pred.ndim != 2:

​        raise ValueError("y_pred must be shape (N, out_dim).")

​    _, out_dim = y_pred.shape



​    fig, axes = plt.subplots(1, out_dim, figsize=(4*out_dim, 4))



​    if out_dim == 1:

​        axes = [axes]



​    for i in range(out_dim):

​        errors = (y_true[:, i] - y_pred[:, i])**2

​        r2_val = r2_score(y_true[:, i], y_pred[:, i])

​        ax = axes[i]



​        sc = ax.scatter(

​            y_true[:, i],

​            y_pred[:, i],

​            c=errors,

​            alpha=0.5,

​            cmap='brg'

​        )

​        min_val = min(y_true[:, i].min(), y_pred[:, i].min())

​        max_val = max(y_true[:, i].max(), y_pred[:, i].max())

​        ax.plot([min_val, max_val],

​                [min_val, max_val],

​                'r--', linewidth=1.5)



​        if y_labels and i < len(y_labels):

​            ax.set_title(f"{y_labels[i]} (MSE)\nR²={r2_val:.3f}")

​            ax.set_xlabel(f"True {y_labels[i]}")

​            ax.set_ylabel(f"Pred {y_labels[i]}")

​        else:

​            ax.set_title(f"Output {i} (MSE)\nR²={r2_val:.3f}")

​            ax.set_xlabel("True")

​            ax.set_ylabel("Pred")



​        cbar = fig.colorbar(sc, ax=ax)

​        cbar.set_label("Squared Error")



​    fig.subplots_adjust(left=0.07, right=0.95, wspace=0.4, top=0.88, bottom=0.12)

​    plt.savefig(filename, dpi=700)

​    plt.close()



def plot_scatter_3d_outputs_mae(y_true, y_pred, y_labels=None,

​                                filename="scatter_3d_mae.jpg"):

​    """

​    类似上面, 颜色=MAE

​    """

​    ensure_dir_for_file(filename)

​    if y_pred.ndim != 2:

​        raise ValueError("y_pred must be shape (N, out_dim).")

​    _, out_dim = y_pred.shape



​    fig, axes = plt.subplots(1, out_dim, figsize=(4*out_dim, 4))



​    if out_dim == 1:

​        axes = [axes]



​    for i in range(out_dim):

​        errors = np.abs(y_true[:, i] - y_pred[:, i])

​        r2_val = r2_score(y_true[:, i], y_pred[:, i])

​        ax = axes[i]



​        sc = ax.scatter(

​            y_true[:, i],

​            y_pred[:, i],

​            c=errors,

​            alpha=0.5,

​            cmap='ocean'

​        )

​        min_val = min(y_true[:, i].min(), y_pred[:, i].min())

​        max_val = max(y_true[:, i].max(), y_pred[:, i].max())

​        ax.plot([min_val, max_val],

​                [min_val, max_val],

​                'r--', linewidth=1.5)



​        if y_labels and i < len(y_labels):

​            ax.set_title(f"{y_labels[i]} (MAE)\nR²={r2_val:.3f}")

​            ax.set_xlabel(f"True {y_labels[i]}")

​            ax.set_ylabel(f"Pred {y_labels[i]}")

​        else:

​            ax.set_title(f"Output {i} (MAE)\nR²={r2_val:.3f}")

​            ax.set_xlabel("True")

​            ax.set_ylabel("Pred")



​        cbar = fig.colorbar(sc, ax=ax)

​        cbar.set_label("Absolute Error")



​    fig.subplots_adjust(left=0.07, right=0.95, wspace=0.4, top=0.88, bottom=0.12)

​    plt.savefig(filename, dpi=700)

​    plt.close()



def plot_residual_histogram(y_true, y_pred, y_labels=None,

​                            cmap_name="coolwarm", vmin=-45, vmax=45,

​                            filename="residual_hist.jpg"):

​    """

​    每个输出画一个直方图, 并对每个bin以颜色标识 residual

​    """

​    ensure_dir_for_file(filename)

​    residuals = y_true - y_pred

​    n_out = residuals.shape[1]



​    fig, axes = plt.subplots(1, n_out, figsize=(4*n_out, 4.5))

​    if n_out == 1:

​        axes = [axes]



​    bins_array = np.linspace(vmin, vmax, 31)

​    cmap = cm.get_cmap(cmap_name)

​    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)



​    for i in range(n_out):

​        ax = axes[i]

​        hist_data, bin_edges, patches = ax.hist(

​            residuals[:, i],

​            bins=bins_array,

​            alpha=0.9,

​            edgecolor='none'

​        )

​        for b_idx, patch in enumerate(patches):

​            bin_center = 0.5*(bin_edges[b_idx] + bin_edges[b_idx+1])

​            patch.set_facecolor(cmap(norm(bin_center)))



​        \# 设置标题或标签

​        if y_labels and i < len(y_labels):

​            ax.set_title(f"Residual of {y_labels[i]}")

​        else:

​            ax.set_title(f"Output {i} Residual")



​        ax.set_xlabel("Residual")

​        ax.set_ylabel("Count")

​        ax.set_xlim(vmin, vmax)



​    \# 统一 colorbar

​    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

​    sm.set_array([])  # 只用于colorbar

​    cbar = fig.colorbar(

​        sm, ax=axes,

​        orientation="horizontal",

​        fraction=0.07,

​        pad=0.20,

​        shrink=0.9

​    )

​    cbar.set_label("Residual Value")



​    \# 不用 tight_layout，自己调子图位置

​    fig.subplots_adjust(left=0.08, right=0.95, top=0.82, bottom=0.2, wspace=0.3)



​    plt.savefig(filename, dpi=300)

​    plt.close()



def plot_residual_kde(y_true, y_pred, y_labels=None,

​                      cmap_name="coolwarm", vmin=-45, vmax=45,

​                      filename="residual_kde.jpg"):

​    """

​    每个输出画一个KDE, 并以填充色块来显示 residual

​    """

​    ensure_dir_for_file(filename)

​    residuals = y_true - y_pred

​    n_out = residuals.shape[1]



​    fig, axes = plt.subplots(1, n_out, figsize=(4*n_out, 4.5))

​    if n_out == 1:

​        axes = [axes]



​    cmap = cm.get_cmap(cmap_name)

​    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)



​    for i in range(n_out):

​        ax = axes[i]

​        \# 让 kdeplot 的 x 范围限于 [vmin, vmax] 并 fill=False

​        sns.kdeplot(

​            residuals[:, i],

​            ax=ax,

​            fill=False,

​            color="black",

​            clip=(vmin, vmax)

​        )

​        lines = ax.get_lines()

​        if not lines:

​            continue

​        line = lines[-1]

​        x_plot = line.get_xdata()

​        y_plot = line.get_ydata()



​        idxsort = np.argsort(x_plot)

​        x_plot = x_plot[idxsort]

​        y_plot = y_plot[idxsort]



​        \# 逐小区间涂色

​        for j in range(len(x_plot)-1):

​            x0, x1 = x_plot[j], x_plot[j+1]

​            y0, y1 = y_plot[j], y_plot[j+1]

​            \# 取中点的 residual 来决定颜色

​            xmid = 0.5*(x0 + x1)

​            color = cmap(norm(xmid))

​            verts = np.array([

​                [x0, 0],

​                [x0, y0],

​                [x1, y1],

​                [x1, 0]

​            ])

​            poly = PolyCollection([verts], facecolors=[color], edgecolor='none', alpha=0.6)

​            ax.add_collection(poly)



​        if y_labels and i < len(y_labels):

​            ax.set_title(f"Residual KDE of {y_labels[i]}")

​        else:

​            ax.set_title(f"KDE - Out {i}")



​        ax.set_xlabel("Residual")

​        ax.set_ylabel("Density")

​        ax.set_xlim(vmin, vmax)



​    \# 统一 colorbar

​    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

​    sm.set_array([])

​    cbar = fig.colorbar(

​        sm, ax=axes,

​        orientation="horizontal",

​        fraction=0.07,

​        pad=0.20,

​        shrink=0.9

​    )

​    cbar.set_label("Residual Value")



​    \# 手动调位置

​    fig.subplots_adjust(left=0.08, right=0.95, top=0.82, bottom=0.2, wspace=0.3)



​    plt.savefig(filename, dpi=300)

​    plt.close()



def plot_rf_feature_importance_bar(rf_model, feature_names,

​                                   filename, top_k=20, threshold=0.05):

​    """

​    RF特征重要度

​    """

​    ensure_dir_for_file(filename)

​    importances = rf_model.feature_importances_

​    if importances is None or len(importances) == 0:

​        print("[WARN] No importances found!")

​        return



​    sorted_idx = np.argsort(importances)[::-1]

​    top_idx = sorted_idx[:top_k]

​    top_feats = [feature_names[i] for i in top_idx]

​    top_imps = importances[top_idx]

​    colors = ["red" if imp > threshold else "blue" for imp in top_imps]



​    fig, ax = plt.subplots(figsize=(8,6))

​    ax.barh(range(len(top_imps)), top_imps, align='center', color=colors)

​    ax.set_yticks(range(len(top_imps)))

​    ax.set_yticklabels(top_feats, fontsize=10)

​    ax.invert_yaxis()

​    ax.set_xlabel("Feature Importance", fontsize=12)

​    ax.set_title(f"Feature Importance (Top-{top_k})", fontsize=14, fontweight='bold')

​    ax.axvspan(0, threshold, facecolor='lightgray', alpha=0.5)

​    ax.axvline(threshold, color='red', linestyle='dashed', linewidth=2)



​    legend_e = [

​        Patch(facecolor="red", label=f"> {threshold}"),

​        Patch(facecolor="royalblue", label=f"<= {threshold}")

​    ]

​    ax.legend(handles=legend_e, loc="lower right", fontsize=12)



​    fig.subplots_adjust(left=0.3, right=0.95, top=0.88, bottom=0.1)

​    plt.savefig(filename, dpi=700)

​    plt.close()





\# ------------------- 三指标条形图 + 过拟合条形图 -------------------

def plot_three_metrics_horizontal(metrics_data, save_name="three_metrics.jpg"):

​    """

​    metrics_data[mtype] = {"MSE":..., "MAE":..., "R2":...}

​    """

​    ensure_dir_for_file(save_name)

​    model_names = list(metrics_data.keys())

​    mse_vals = [metrics_data[m]["MSE"] for m in model_names]

​    mae_vals = [metrics_data[m]["MAE"] for m in model_names]

​    r2_vals  = [metrics_data[m]["R2"]  for m in model_names]



​    fig, axes = plt.subplots(1,3, figsize=(16,5))



​    def plot_hbar_with_mean(ax, model_names, values, subplot_label, metric_label, bigger_is_better=False):

​        arr = np.array(values)

​        if bigger_is_better:

​            best_idx = arr.argmax()

​            worst_idx = arr.argmin()

​        else:

​            best_idx = arr.argmin()

​            worst_idx = arr.argmax()



​        colors = []

​        for i,vv in enumerate(arr):

​            if i == best_idx:

​                colors.append("red")

​            elif i == worst_idx:

​                colors.append("blue")

​            else:

​                colors.append("green")



​        mean_val = arr.mean()

​        y_positions = np.arange(len(arr))

​        ax.barh(y_positions, arr, color=colors, alpha=0.8, height=0.4)

​        ax.set_yticks(y_positions)

​        ax.set_yticklabels(model_names)

​        ax.invert_yaxis()

​        ax.text(-0.08,1.05, subplot_label, transform=ax.transAxes,

​                ha="left", va="top", fontsize=14, fontweight="bold")

​        ax.set_title(metric_label, fontsize=14, fontweight='bold')



​        for i,vv in enumerate(arr):

​            ax.text(vv, i, f"{vv:.2f}", ha=("left" if vv>=0 else "right"),

​                    va="center", fontsize=10)



​        ax.axvline(mean_val, color='gray', linestyle='--', linewidth=2)

​        xmin, xmax = sorted([0, mean_val])

​        ax.axvspan(xmin, xmax, facecolor='gray', alpha=0.2)



​        max_val = arr.max()

​        min_val = arr.min()

​        if min_val < 0:

​            ax.set_xlim(min_val*1.1, max_val*1.79)

​        else:

​            ax.set_xlim(0, max_val*1.79)



​        legend_e = [

​            Patch(facecolor="red", label="Best"),

​            Patch(facecolor="blue", label="Worst"),

​            Patch(facecolor="green", label="Ordinary"),

​            Patch(facecolor="gray", alpha=0.2, label="Under Mean"),

​        ]

​        ax.legend(handles=legend_e, loc="lower right")



​    plot_hbar_with_mean(axes[0], model_names, mse_vals, "(a)", "MSE (Lower=Better)", bigger_is_better=False)

​    plot_hbar_with_mean(axes[1], model_names, mae_vals, "(b)", "MAE (Lower=Better)", bigger_is_better=False)

​    plot_hbar_with_mean(axes[2], model_names, r2_vals,  "(c)", "R2 (Higher=Better)", bigger_is_better=True)



​    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.1, wspace=0.3)

​    plt.savefig(save_name, dpi=300)

​    plt.close()

​    print(f"[plot_three_metrics_horizontal] => {save_name}")



def plot_overfitting_horizontal(overfit_data, save_name="overfitting_horizontal.jpg"):

​    """

​    overfit_data[mtype] = {"MSE_ratio":..., "R2_diff":...}

​    """

​    ensure_dir_for_file(save_name)

​    model_names = list(overfit_data.keys())

​    msr_vals = [overfit_data[m]["MSE_ratio"] for m in model_names]

​    r2d_vals = [overfit_data[m]["R2_diff"]   for m in model_names]



​    fig, axes = plt.subplots(1,2, figsize=(12,5))



​    def plot_hbar_threshold(ax, model_names, values, subplot_label, metric_label,

​                            bigger_is_better=False, threshold_h=0.5, threshold_l=0.0):

​        arr= np.array(values)

​        if bigger_is_better:

​            best_idx= arr.argmax()

​            worst_idx= arr.argmin()

​        else:

​            best_idx= arr.argmin()

​            worst_idx= arr.argmax()



​        colors= []

​        for i,vv in enumerate(arr):

​            if i==best_idx:

​                colors.append("red")

​            elif i==worst_idx:

​                colors.append("blue")

​            else:

​                colors.append("green")



​        y_positions= np.arange(len(arr))

​        ax.barh(y_positions, arr, color=colors, alpha=0.8, height=0.4)

​        ax.set_yticks(y_positions)

​        ax.set_yticklabels(model_names)

​        ax.invert_yaxis()

​        ax.text(-0.08,1.05, subplot_label, transform=ax.transAxes,

​                ha="left", va="top", fontsize=14, fontweight="bold")

​        ax.set_title(metric_label, fontsize=14, fontweight='bold')



​        for i,vv in enumerate(arr):

​            ax.text(vv, i, f"{vv:.2f}", ha=("left" if vv>=0 else "right"),

​                    va="center", fontsize=10)



​        if threshold_l==0.0:

​            ax.axvspan(threshold_l, threshold_h, facecolor='gray', alpha=0.2)

​            legend_e= [ Patch(facecolor='gray', alpha=0.2, label="Acceptable") ]

​        else:

​            ax.axvspan(0, threshold_l, facecolor='gray', alpha=0.2)

​            ax.axvspan(threshold_l, threshold_h, facecolor='lightcoral', alpha=0.3)

​            ax.axvline(threshold_l, color='gray', linestyle='--', linewidth=2)

​            ax.axvline(threshold_h, color='gray', linestyle='--', linewidth=2)

​            legend_e= [

​                Patch(facecolor='gray', alpha=0.2, label="Acceptable"),

​                Patch(facecolor='lightcoral', alpha=0.3, label="Overfitting Risk")

​            ]



​        max_val= arr.max()

​        min_val= arr.min()

​        if min_val<0:

​            ax.set_xlim(min_val*1.1, max_val*1.79)

​        else:

​            ax.set_xlim(0, max_val*1.79)



​        legend_e.extend([

​            Patch(facecolor="red", label="Best"),

​            Patch(facecolor="blue", label="Worst"),

​            Patch(facecolor="green", label="Ordinary")

​        ])

​        ax.legend(handles=legend_e, loc="lower right")



​    plot_hbar_threshold(axes[0],

​                        model_names, msr_vals,

​                        "(a)",

​                        "MSE Ratio (Val/Train)\n(Lower=Better)",

​                        bigger_is_better=False,

​                        threshold_h=10)



​    plot_hbar_threshold(axes[1],

​                        model_names, r2d_vals,

​                        "(b)",

​                        "R2 diff (Train - Val)\n(Lower=Better)",

​                        bigger_is_better=False,

​                        threshold_h=0.2,

​                        threshold_l=0.15)



​    fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.1, wspace=0.3)

​    plt.savefig(save_name, dpi=300)

​    plt.close()

​    print(f"[plot_overfitting_horizontal] => {save_name}")





\# ------------------- 原始数据分析(kde, scatter, boxplot...) -------------------

def plot_kde_distribution(df, columns, filename):

​    """

​    绘制多个列的KDE分布，每个子图一个colorbar + 用clip避免越界

​    """

​    ensure_dir_for_file(filename)

​    fig, axes= plt.subplots(1,len(columns), figsize=(5*len(columns),5))

​    if len(columns)==1:

​        axes= [axes]



​    for i,col in enumerate(columns):

​        ax= axes[i]

​        if col not in df.columns:

​            ax.text(0.5,0.5,f"'{col}' not in df", ha='center', va='center')

​            continue



​        \# 使用 clip=(df[col].min(), df[col].max()) 保持与原版一致

​        sns.kdeplot(df[col], ax=ax, fill=False, color='black',

​                    clip=(df[col].min(), df[col].max()))

​        lines= ax.get_lines()

​        if not lines:

​            ax.set_title(f"No Data for {col}")

​            continue



​        line= lines[-1]

​        x_plot, y_plot= line.get_xdata(), line.get_ydata()

​        idxsort= np.argsort(x_plot)

​        x_plot, y_plot= x_plot[idxsort], y_plot[idxsort]



​        vmin= df[col].min()

​        vmax= df[col].max()

​        cmap= cm.get_cmap("coolwarm")

​        norm= mcolors.Normalize(vmin=vmin, vmax=vmax)



​        \# 为了让KDE颜色区分明显，每个小区间用多边形填充

​        for j in range(len(x_plot)-1):

​            x0, x1= x_plot[j], x_plot[j+1]

​            y0, y1= y_plot[j], y_plot[j+1]

​            color= cmap(norm((x0+x1)*0.5))

​            verts= np.array([

​                [x0,0],

​                [x0,y0],

​                [x1,y1],

​                [x1,0]

​            ])

​            poly= PolyCollection([verts], facecolors=[color], edgecolor='none', alpha=0.6)

​            ax.add_collection(poly)



​        ax.set_title(f"KDE of {col}")

​        ax.set_xlabel(col)

​        ax.set_ylabel("Density")

​        ax.set_xlim(vmin, vmax)



​        sm= cm.ScalarMappable(norm=norm, cmap=cmap)

​        sm.set_array([])

​        cb= plt.colorbar(sm, ax=ax)

​        cb.set_label("Value Range", fontweight='bold', fontsize=12)

​        cb.ax.tick_params(labelsize=10)



​    fig.subplots_adjust(left=0.06, right=0.96, top=0.9, bottom=0.15, wspace=0.4)

​    plt.savefig(filename, dpi=300)

​    plt.close()



def plot_catalyst_size_vs_product(df, filename):

​    ensure_dir_for_file(filename)

​    fig, axes= plt.subplots(2,2, figsize=(15,10), sharex=True)

​    products= ['H2','CO','C1','C2+']

​    for i,product in enumerate(products):

​        ax= axes[i//2, i%2]

​        needed= ['Particle size (nm)','Active metal',product]

​        if all(c in df.columns for c in needed):

​            sns.scatterplot(x='Particle size (nm)', y=product, hue='Active metal',

​                            data=df, ax=ax, alpha=0.7)

​            ax.set_title(f'Particle size vs {product} Yield')

​            ax.set_xlabel('Particle size (nm)')

​            ax.set_ylabel(f'{product} Yield (%)')

​        else:

​            ax.text(0.5,0.5,f"Cols not found => {product}", ha='center', va='center')



​    fig.subplots_adjust(left=0.08, right=0.95, wspace=0.3, hspace=0.3, top=0.92, bottom=0.08)

​    plt.savefig(filename, dpi=300)

​    plt.close()



def plot_potential_vs_product_by_electrolyte(df, filename):

​    ensure_dir_for_file(filename)

​    fig, axes= plt.subplots(2,2, figsize=(15,10), sharex=True)

​    products= ['H2','CO','C1','C2+']

​    for i,product in enumerate(products):

​        ax= axes[i//2, i%2]

​        needed= ['Potential (V vs. RHE)','Electrode support',product]

​        if all(c in df.columns for c in needed):

​            sns.scatterplot(x='Potential (V vs. RHE)', y=product,

​                            hue='Electrode support',

​                            data=df, ax=ax, alpha=0.7)

​            ax.set_title(f'Potential vs {product}')

​            ax.set_xlabel('Potential (V vs. RHE)')

​            ax.set_ylabel(f'{product} Yield (%)')

​        else:

​            ax.text(0.5,0.5,f"Cols not found => {product}", ha='center', va='center')



​    fig.subplots_adjust(left=0.08, right=0.95, wspace=0.3, hspace=0.3, top=0.92, bottom=0.08)

​    plt.savefig(filename, dpi=300)

​    plt.close()



def plot_product_distribution_by_catalyst_and_potential(df, filename):

​    ensure_dir_for_file(filename)

​    fig, axes= plt.subplots(1,4, figsize=(20,5))

​    products= ['H2','CO','C1','C2+']

​    if 'Potential (V vs. RHE)' in df.columns:

​        df['Potential_bin']= pd.cut(df['Potential (V vs. RHE)'], bins=5)

​    else:

​        df['Potential_bin']= "Unknown"

​    for i,product in enumerate(products):

​        ax= axes[i]

​        needed= ['Active metal', product, 'Potential_bin']

​        if all(c in df.columns for c in needed):

​            sns.boxplot(x='Active metal', y=product,

​                        hue='Potential_bin', data=df, ax=ax)

​            ax.set_title(f'{product} by ActiveMetal & Potential')

​            ax.tick_params(axis='x', rotation=45)

​        else:

​            ax.text(0.5,0.5,f"Cols not found => {product}", ha='center', va='center')

​    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.3, top=0.85, bottom=0.15)

​    plt.savefig(filename, dpi=300)

​    plt.close()



def plot_product_vs_potential_bin(df, filename):

​    ensure_dir_for_file(filename)

​    products= ['H2','CO','C1','C2+']

​    if 'Potential (V vs. RHE)' not in df.columns:

​        print("[WARN] no Potential => skip")

​        return

​    df['Potential_bin_custom']= pd.cut(df['Potential (V vs. RHE)'], bins=5)

​    fig, axes= plt.subplots(1,4, figsize=(20,5))

​    for i,product in enumerate(products):

​        ax= axes[i]

​        if product not in df.columns:

​            ax.text(0.5,0.5,f"No col => {product}", ha='center', va='center')

​            continue

​        sns.boxplot(x='Potential_bin_custom', y=product, data=df, ax=ax)

​        ax.set_title(f"{product} vs Potential Bin")

​        ax.tick_params(axis='x', rotation=45)

​    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.3, top=0.85, bottom=0.15)

​    plt.savefig(filename, dpi=300)

​    plt.close()



def plot_product_vs_shape(df, filename):

​    ensure_dir_for_file(filename)

​    products= ['H2','CO','C1','C2+']

​    if 'Shape' not in df.columns:

​        print("[WARN] no 'Shape' => skip")

​        return

​    fig, axes= plt.subplots(1,4, figsize=(20,5))

​    for i,product in enumerate(products):

​        ax= axes[i]

​        if product not in df.columns:

​            ax.text(0.5,0.5,f"No col => {product}", ha='center', va='center')

​            continue

​        sns.boxplot(x='Shape', y=product, data=df, ax=ax)

​        ax.set_title(f"{product} vs Shape")

​        ax.tick_params(axis='x', rotation=45)

​    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.3, top=0.85, bottom=0.15)

​    plt.savefig(filename, dpi=300)

​    plt.close()



def plot_product_vs_catalyst(df, filename):

​    ensure_dir_for_file(filename)

​    products= ['H2','CO','C1','C2+']

​    if 'Active metal' not in df.columns:

​        print("[WARN] no 'Active metal' => skip")

​        return

​    fig, axes= plt.subplots(1,4, figsize=(20,5))

​    for i,product in enumerate(products):

​        ax= axes[i]

​        if product not in df.columns:

​            ax.text(0.5,0.5,f"No col => {product}", ha='center', va='center')

​            continue

​        sns.boxplot(x='Active metal', y=product, data=df, ax=ax)

​        ax.set_title(f"{product} vs Active metal")

​        ax.tick_params(axis='x', rotation=45)

​    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.3, top=0.85, bottom=0.15)

​    plt.savefig(filename, dpi=300)

​    plt.close()



def plot_potential_vs_product(df, filename):

​    ensure_dir_for_file(filename)

​    products= ['H2','CO','C1','C2+']

​    if 'Potential (V vs. RHE)' not in df.columns:

​        print("[WARN] no 'Potential (V vs. RHE)' => skip")

​        return

​    plt.figure(figsize=(7,6))

​    for product in products:

​        if product in df.columns:

​            plt.scatter(df['Potential (V vs. RHE)'], df[product], label=product, alpha=0.7)

​    plt.title("Potential vs Products")

​    plt.xlabel("Potential (V vs. RHE)")

​    plt.ylabel("Yield (%)")

​    plt.legend()

​    plt.tight_layout()

​    plt.savefig(filename, dpi=300)

​    plt.close()



\# ------------------- 推理可视化(2D Heatmap + Confusion) -------------------

def plot_2d_heatmap_from_npy(grid_x, grid_y, heatmap_pred,

​                             out_dir,

​                             y_col_names=None,

​                             colorbar_extend_ratio=0.25):

​    """

​    heatmap_pred shape=(H,W,out_dim)

​    each output => separate figure

​    colorbar范围 => [max(0, min_val*(1-extend)), min(100, max_val*(1+extend))]

​    """

​    ensure_dir(out_dir)

​    H,W,out_dim= heatmap_pred.shape

​    for odx in range(out_dim):

​        z_= heatmap_pred[:,:,odx]

​        real_min= z_.min()

​        real_max= z_.max()

​        vmin_= max(0, real_min*(1-colorbar_extend_ratio))

​        vmax_= min(100, real_max*(1+colorbar_extend_ratio))



​        fig, ax = plt.subplots(figsize=(6,5))

​        norm_= mcolors.Normalize(vmin=vmin_, vmax=vmax_)

​        mesh_ = ax.pcolormesh(grid_x, grid_y, z_, shading='auto', cmap='viridis', norm=norm_)

​        cb_ = plt.colorbar(mesh_, ax=ax)

​        if y_col_names and odx<len(y_col_names):

​            cb_.set_label(y_col_names[odx], fontsize=12)

​            ax.set_title(f"2D Heatmap of {y_col_names[odx]}")

​        else:

​            cb_.set_label(f"Output_{odx}", fontsize=12)

​            ax.set_title(f"2D Heatmap - out {odx}")



​        ax.set_xlabel("X-axis", fontsize=12)

​        ax.set_ylabel("Y-axis", fontsize=12)



​        fig.subplots_adjust(left=0.15, right=0.92, top=0.88, bottom=0.15)

​        out_png= os.path.join(out_dir, f"heatmap_output_{odx+1}.png")

​        plt.savefig(out_png, dpi=150)

​        plt.close()



def plot_confusion_from_npy(confusion_pred, row_labels, col_labels,

​                            out_dir,

​                            y_col_names=None,

​                            cell_scale=1/3,

​                            colorbar_extend_ratio=0.25):

​    """

​    confusion_pred shape=(n_rows,n_cols,out_dim)

​    每格分4三角, 在三角内显示 val.

​    colorbar范围 => [max(0, min_val*(1-extend)), min(100, max_val*(1+extend))]

​    保持正方形外观 + 字体稍大 + colorbar.

​    """

​    ensure_dir(out_dir)

​    n_rows, n_cols, out_dim= confusion_pred.shape



​    allvals= confusion_pred.reshape(-1, out_dim)

​    mins_= allvals.min(axis=0)

​    maxs_= allvals.max(axis=0)



​    fig, ax= plt.subplots(figsize=(10,10))

​    ax.set_title("Confusion-like MIMO (Show numeric)", fontsize=14)

​    ax.set_aspect("equal", "box")



​    \# 画网格线

​    for rr in range(n_rows+1):

​        ax.axhline(rr*cell_scale, color='black', linewidth=1)

​    for cc in range(n_cols+1):

​        ax.axvline(cc*cell_scale, color='black', linewidth=1)



​    \# 最多4个输出 => 4种 cmap

​    cmaps= [plt.get_cmap("Reds"), plt.get_cmap("Blues"),

​            plt.get_cmap("Greens"), plt.get_cmap("Oranges")]



​    for i in range(n_rows):

​        for j in range(n_cols):

​            vals= confusion_pred[i,j,:]

​            \# corners

​            BL= (j*cell_scale, i*cell_scale)

​            BR= ((j+1)*cell_scale, i*cell_scale)

​            TL= (j*cell_scale, (i+1)*cell_scale)

​            TR= ((j+1)*cell_scale, (i+1)*cell_scale)

​            Cx= j*cell_scale + cell_scale/2

​            Cy= i*cell_scale + cell_scale/2



​            for odx in range(min(4,out_dim)):

​                val_= vals[odx]

​                real_min= mins_[odx]

​                real_max= maxs_[odx]

​                vmin_= max(0, real_min*(1-colorbar_extend_ratio))

​                vmax_= min(100, real_max*(1+colorbar_extend_ratio))

​                norm_= matplotlib.colors.Normalize(vmin=vmin_, vmax=vmax_)

​                color_= cmaps[odx](norm_(val_))



​                \# 4个输出 => 4个三角区域

​                if odx==0:

​                    poly= [TL, (Cx,Cy), TR]

​                    tx= (TL[0]+TR[0]+Cx)/3

​                    ty= (TL[1]+TR[1]+Cy)/3

​                elif odx==1:

​                    poly= [TR, (Cx,Cy), BR]

​                    tx= (TR[0]+BR[0]+Cx)/3

​                    ty= (TR[1]+BR[1]+Cy)/3

​                elif odx==2:

​                    poly= [BR, (Cx,Cy), BL]

​                    tx= (BR[0]+BL[0]+Cx)/3

​                    ty= (BR[1]+BL[1]+Cy)/3

​                else:

​                    poly= [BL, (Cx,Cy), TL]

​                    tx= (BL[0]+TL[0]+Cx)/3

​                    ty= (BL[1]+TL[1]+Cy)/3



​                ax.add_patch(plt.Polygon(poly, facecolor=color_, alpha=0.9))

​                \# 在三角形中写数值

​                ax.text(tx, ty, f"{val_:.1f}",

​                        ha='center', va='center',

​                        fontsize=10, color='black')



​    ax.set_xlim(0, n_cols*cell_scale)

​    ax.set_ylim(0, n_rows*cell_scale)

​    ax.invert_yaxis()



​    \# 设置坐标轴标签

​    ax.set_xticks([(j+0.5)*cell_scale for j in range(n_cols)])

​    ax.set_yticks([(i+0.5)*cell_scale for i in range(n_rows)])

​    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=11)

​    ax.set_yticklabels(row_labels, fontsize=11)



​    \# 手动调大右边空白, 用于放4个 colorbar

​    fig.subplots_adjust(left=0.1, right=0.68, top=0.9, bottom=0.1)



​    \# 画4个 colorbar

​    for odx in range(min(4,out_dim)):

​        real_min= mins_[odx]

​        real_max= maxs_[odx]

​        vmin_= max(0, real_min*(1-colorbar_extend_ratio))

​        vmax_= min(100, real_max*(1+colorbar_extend_ratio))

​        norm_= matplotlib.colors.Normalize(vmin=vmin_, vmax=vmax_)

​        cax_left= 0.7 + 0.07*odx

​        cax= fig.add_axes([cax_left, 0.1, 0.03, 0.8])

​        cb_= matplotlib.colorbar.ColorbarBase(cax, cmap=cmaps[odx], norm=norm_)

​        if y_col_names and odx<len(y_col_names):

​            label_= y_col_names[odx]

​        else:

​            label_= f"Out {odx}"

​        cb_.set_label(label_, fontsize=11)

​        cb_.ax.tick_params(labelsize=10)



​    outfn= os.path.join(out_dir, "confusion_matrix_mimo.png")

​    plt.savefig(outfn, dpi=150)

​    plt.close()

​    print(f"[INFO] Confusion => {outfn}")





"""

visualization.py



需求:

1) 从 postprocessing/<csv_name>/train 读取:

   \- df_raw_14.csv => 做 correlation (普通) + DataAnalysis

   \- X_onehot.npy => 做 correlation_heatmap_one_hot

   \- Y_train.npy, Y_val.npy => 用于画散点/残差

   \- 对每个模型 => 读取 train_pred.npy, val_pred.npy, metrics.pkl, train_losses.npy, val_losses.npy

​     => 画散点、残差、MAE、MSE、Loss曲线、FeatureImportance(若有)

2) 从 postprocessing/<csv_name>/inference/<model_type> 读取:

   \- heatmap_pred.npy, grid_x.npy, grid_y.npy => 2D heatmap

   \- confusion_pred.npy => confusion-like

3) 输出图到 ./evaluation/figures/<csv_name>/...
4) 已去掉 K-Fold 逻辑.

"""



import os

import yaml

import numpy as np

import pandas as pd

import joblib



from utils import (

​    ensure_dir,

​    ensure_dir_for_file,

​    \# correlation

​    plot_correlation_heatmap,

​    \# data analysis

​    plot_kde_distribution,

​    plot_catalyst_size_vs_product,

​    plot_potential_vs_product_by_electrolyte,

​    plot_product_distribution_by_catalyst_and_potential,

​    plot_product_vs_potential_bin,

​    plot_product_vs_shape,

​    plot_product_vs_catalyst,

​    plot_potential_vs_product,

​    \# model metrics

​    plot_three_metrics_horizontal,

​    plot_overfitting_horizontal,

​    plot_loss_curve,

​    plot_scatter_3d_outputs_mse,

​    plot_scatter_3d_outputs_mae,

​    plot_residual_histogram,

​    plot_residual_kde,

​    plot_rf_feature_importance_bar,

​    \# inference

​    plot_2d_heatmap_from_npy,

​    plot_confusion_from_npy

)



def visualize_main():

​    \# 读取配置文件

​    with open("./configs/config.yaml", "r") as f:

​        config = yaml.safe_load(f)



​    csv_path = config["data"]["path"]

​    csv_name = os.path.splitext(os.path.basename(csv_path))[0]



​    \# 1) 读取 postprocessing/<csv_name>/train 数据

​    base_train = os.path.join("postprocessing", csv_name, "train")

​    if not os.path.isdir(base_train):

​        print(f"[WARN] train folder not found => {base_train}")

​        return



​    \# ========= 1.1) 读取 df_raw_14.csv ==========

​    raw_csv_path = os.path.join(base_train, "df_raw_14.csv")

​    data_corr_dir = os.path.join("./evaluation/figures", csv_name, "DataCorrelation")

​    ensure_dir(data_corr_dir)



​    if os.path.exists(raw_csv_path):

​        df_raw_14 = pd.read_csv(raw_csv_path)

​        \# 1.1.1) correlation (原始数据)

​        if config["evaluation"].get("save_correlation", False):

​            raw_cols = df_raw_14.columns.tolist()

​            numeric_14 = df_raw_14.select_dtypes(include=[np.number]).columns.tolist()

​            numeric_idx_14 = [raw_cols.index(c) for c in numeric_14]

​            fn1 = os.path.join(data_corr_dir, "correlation_heatmap.jpg")

​            plot_correlation_heatmap(

​                df_raw_14.values,

​                col_names=raw_cols,

​                numeric_cols_idx=numeric_idx_14,

​                filename=fn1

​            )

​        \# 1.1.2) 数据分析图

​        if config["evaluation"].get("save_data_analysis_plots", False):

​            possible_cols = ["Potential (V vs. RHE)", "H2", "CO", "C1", "C2+", "Particle size (nm)"]

​            existing_cols = [c for c in possible_cols if c in df_raw_14.columns]

​            if existing_cols:

​                out_kde = os.path.join(data_corr_dir, "kde_distribution.jpg")

​                plot_kde_distribution(df_raw_14, existing_cols, filename=out_kde)

​            out_cat = os.path.join(data_corr_dir, "catalyst_size_vs_product.jpg")

​            plot_catalyst_size_vs_product(df_raw_14, filename=out_cat)

​            out_pp = os.path.join(data_corr_dir, "potential_vs_product_by_electrolyte.jpg")

​            plot_potential_vs_product_by_electrolyte(df_raw_14, filename=out_pp)

​            out_prod = os.path.join(data_corr_dir, "product_distribution.jpg")

​            plot_product_distribution_by_catalyst_and_potential(df_raw_14, filename=out_prod)

​            out_box_pot = os.path.join(data_corr_dir, "box_product_vs_potential_bin.jpg")

​            plot_product_vs_potential_bin(df_raw_14, filename=out_box_pot)

​            out_box_shape = os.path.join(data_corr_dir, "box_product_vs_shape.jpg")

​            plot_product_vs_shape(df_raw_14, filename=out_box_shape)

​            out_box_cat = os.path.join(data_corr_dir, "box_product_vs_catalyst.jpg")

​            plot_product_vs_catalyst(df_raw_14, filename=out_box_cat)

​            out_three = os.path.join(data_corr_dir, "three_dot_potential_vs_product.jpg")

​            plot_potential_vs_product(df_raw_14, filename=out_three)

​    else:

​        print(f"[WARN] df_raw_14.csv not found => {raw_csv_path}")



​    \# ========= 1.2) 读取 onehot 数据, 做 correlation ==========

​    x_onehot_path = os.path.join(base_train, "X_onehot.npy")

​    if os.path.exists(x_onehot_path) and config["evaluation"].get("save_correlation", False):

​        X_onehot = np.load(x_onehot_path)

​        x_col_names_path = os.path.join(base_train, "x_col_names.npy")

​        if os.path.exists(x_col_names_path):

​            onehot_colnames = list(np.load(x_col_names_path, allow_pickle=True))

​        else:

​            onehot_colnames = [f"Feature {i+1}" for i in range(X_onehot.shape[1])]

​        numeric_idx = list(range(X_onehot.shape[1]))  # onehot 全部当作数值列(0/1)

​        fn2 = os.path.join(data_corr_dir, "correlation_heatmap_one_hot.jpg")

​        plot_correlation_heatmap(

​            X_onehot,

​            col_names=onehot_colnames,

​            numeric_cols_idx=numeric_idx,

​            filename=fn2

​        )



​    \# ========= 1.3) 读取 Y_train.npy, Y_val.npy ==========

​    y_train_path = os.path.join(base_train, "Y_train.npy")

​    y_val_path = os.path.join(base_train, "Y_val.npy")

​    Y_train = np.load(y_train_path) if os.path.exists(y_train_path) else None

​    Y_val = np.load(y_val_path) if os.path.exists(y_val_path) else None



​    \# ========= 1.4) 针对每个模型, 可视化训练/验证结果 ==========

​    model_types = config["model"]["types"]

​    for mtype in model_types:

​        model_subdir = os.path.join(base_train, mtype)

​        if not os.path.isdir(model_subdir):

​            print(f"[WARN] no train folder for model type => {model_subdir}")

​            continue



​        metrics_pkl = os.path.join(model_subdir, "metrics.pkl")

​        train_pred_path = os.path.join(model_subdir, "train_pred.npy")

​        val_pred_path = os.path.join(model_subdir, "val_pred.npy")

​        train_loss_path = os.path.join(model_subdir, "train_losses.npy")

​        val_loss_path = os.path.join(model_subdir, "val_losses.npy")



​        \# 读取 metrics

​        train_metrics = None

​        val_metrics = None

​        if os.path.exists(metrics_pkl):

​            data_ = joblib.load(metrics_pkl)

​            train_metrics = data_.get("train_metrics", None)

​            val_metrics = data_.get("val_metrics", None)

​            print(f"[{mtype}] train_metrics={train_metrics}, val_metrics={val_metrics}")



​        \# 读取预测值

​        train_pred = np.load(train_pred_path) if os.path.exists(train_pred_path) else None

​        val_pred = np.load(val_pred_path) if os.path.exists(val_pred_path) else None



​        \# 读取 Loss

​        train_losses = np.load(train_loss_path) if os.path.exists(train_loss_path) else None

​        val_losses = np.load(val_loss_path) if os.path.exists(val_loss_path) else None



​        \# 读取 y_col_names (若存在, 用于给散点图/残差图传入标签)

​        model_dir = os.path.join("./models", mtype)  # 假设你的训练模型都在 ./models/<mtype> 下

​        ycol_path = os.path.join(model_dir, "y_col_names.npy")

​        if os.path.exists(ycol_path):

​            y_cols = list(np.load(ycol_path, allow_pickle=True))

​        else:

​            y_cols = None



​        model_comp_dir = os.path.join("./evaluation/figures", csv_name, "model_comparison", mtype)

​        ensure_dir(model_comp_dir)



​        \# ---------- 绘制 loss 曲线 ----------

​        if train_losses is not None and val_losses is not None and config["evaluation"].get("save_loss_curve", False):

​            out_lc = os.path.join(model_comp_dir, f"{mtype}_loss_curve.jpg")

​            plot_loss_curve(train_losses, val_losses, filename=out_lc)



​        \# ---------- 绘制散点图 & 残差图 ----------

​        if (Y_train is not None and Y_val is not None) and (train_pred is not None and val_pred is not None):

​            \# scatter MSE

​            if config["evaluation"].get("save_scatter_mse_plot", False):

​                out_mse_tr = os.path.join(model_comp_dir, "full", "train", f"{mtype}_mse_scatter_train.jpg")

​                ensure_dir(os.path.dirname(out_mse_tr))

​                plot_scatter_3d_outputs_mse(Y_train, train_pred, y_labels=y_cols, filename=out_mse_tr)



​                out_mse_val = os.path.join(model_comp_dir, "full", "valid", f"{mtype}_mse_scatter.jpg")

​                ensure_dir(os.path.dirname(out_mse_val))

​                plot_scatter_3d_outputs_mse(Y_val, val_pred, y_labels=y_cols, filename=out_mse_val)



​            \# scatter MAE

​            if config["evaluation"].get("save_scatter_mae_plot", False):

​                out_mae_tr = os.path.join(model_comp_dir, "full", "train", f"{mtype}_mae_scatter_train.jpg")

​                ensure_dir(os.path.dirname(out_mae_tr))

​                plot_scatter_3d_outputs_mae(Y_train, train_pred, y_labels=y_cols, filename=out_mae_tr)



​                out_mae_val = os.path.join(model_comp_dir, "full", "valid", f"{mtype}_mae_scatter.jpg")

​                ensure_dir(os.path.dirname(out_mae_val))

​                plot_scatter_3d_outputs_mae(Y_val, val_pred, y_labels=y_cols, filename=out_mae_val)



​            \# residual hist

​            if config["evaluation"].get("save_residual_hist", False):

​                out_hist_tr = os.path.join(model_comp_dir, "full", "train", f"{mtype}_residual_hist_train.jpg")

​                ensure_dir(os.path.dirname(out_hist_tr))

​                plot_residual_histogram(Y_train, train_pred, y_labels=y_cols, filename=out_hist_tr)



​                out_hist_val = os.path.join(model_comp_dir, "full", "valid", f"{mtype}_residual_hist.jpg")

​                ensure_dir(os.path.dirname(out_hist_val))

​                plot_residual_histogram(Y_val, val_pred, y_labels=y_cols, filename=out_hist_val)



​            \# residual kde

​            if config["evaluation"].get("save_residual_kde", False):

​                out_kde_tr = os.path.join(model_comp_dir, "full", "train", f"{mtype}_residual_kde_train.jpg")

​                ensure_dir(os.path.dirname(out_kde_tr))

​                plot_residual_kde(Y_train, train_pred, y_labels=y_cols, filename=out_kde_tr)



​                out_kde_val = os.path.join(model_comp_dir, "full", "valid", f"{mtype}_residual_kde.jpg")

​                ensure_dir(os.path.dirname(out_kde_val))

​                plot_residual_kde(Y_val, val_pred, y_labels=y_cols, filename=out_kde_val)



​        \# (若有 RF 或其他模型可计算 feature importance，可在此调用 plot_rf_feature_importance_bar)



​    \# ========== 将多个模型的 metrics 汇总到条形图 (three_metrics, overfitting) ==========

​    train_metrics_dict = {}

​    val_metrics_dict = {}

​    for mtype in model_types:

​        mdir = os.path.join(base_train, mtype)

​        mpkl = os.path.join(mdir, "metrics.pkl")

​        if os.path.exists(mpkl):

​            data_ = joblib.load(mpkl)

​            train_metrics_dict[mtype] = data_.get("train_metrics", {})

​            val_metrics_dict[mtype] = data_.get("val_metrics", {})



​    if train_metrics_dict or val_metrics_dict:

​        if train_metrics_dict:

​            out_3train = os.path.join(data_corr_dir, "three_metrics_horizontal_train.jpg")

​            plot_three_metrics_horizontal(train_metrics_dict, save_name=out_3train)

​        if val_metrics_dict:

​            out_3val = os.path.join(data_corr_dir, "three_metrics_horizontal_val.jpg")

​            plot_three_metrics_horizontal(val_metrics_dict, save_name=out_3val)



​        if config["evaluation"].get("save_models_evaluation_bar", False):

​            \# 过拟合对比图

​            if train_metrics_dict and val_metrics_dict:

​                overfit_data = {}

​                for m in train_metrics_dict:

​                    trm = train_metrics_dict[m]

​                    vam = val_metrics_dict[m]

​                    if trm["MSE"] == 0:

​                        ms_ratio = float("inf")

​                    else:

​                        ms_ratio = vam["MSE"] / trm["MSE"]

​                    r2_diff = trm["R2"] - vam["R2"]

​                    overfit_data[m] = {"MSE_ratio": ms_ratio, "R2_diff": r2_diff}

​                out_of = os.path.join(data_corr_dir, "overfitting_single.jpg")

​                plot_overfitting_horizontal(overfit_data, save_name=out_of)



​    \# ========== 推理 (inference) 可视化: Heatmap + Confusion-like ==========

​    base_inf = os.path.join("postprocessing", csv_name, "inference")

​    inf_models = config["inference"].get("models", [])

​    for mtype in inf_models:

​        inf_dir = os.path.join(base_inf, mtype)

​        if not os.path.isdir(inf_dir):

​            print(f"[WARN] no inference dir => {inf_dir}")

​            continue



​        heatmap_path = os.path.join(inf_dir, "heatmap_pred.npy")

​        gridx_path = os.path.join(inf_dir, "grid_x.npy")

​        gridy_path = os.path.join(inf_dir, "grid_y.npy")

​        confusion_path = os.path.join(inf_dir, "confusion_pred.npy")



​        base_out = os.path.join("./evaluation/figures", csv_name, "inference", mtype)

​        ensure_dir(base_out)



​        \# 1) heatmap

​        if os.path.exists(heatmap_path) and os.path.exists(gridx_path) and os.path.exists(gridy_path):

​            heatmap_pred = np.load(heatmap_path)

​            grid_x = np.load(gridx_path)

​            grid_y = np.load(gridy_path)

​            model_dir = os.path.join("./models", mtype)

​            ycol_path = os.path.join(model_dir, "y_col_names.npy")

​            if os.path.exists(ycol_path):

​                y_cols = list(np.load(ycol_path, allow_pickle=True))

​            else:

​                y_cols = None

​            out_hm = os.path.join(base_out, "2d_heatmap")

​            ensure_dir(out_hm)

​            plot_2d_heatmap_from_npy(

​                grid_x, grid_y, heatmap_pred,

​                out_dir=out_hm,

​                y_col_names=y_cols,

​                colorbar_extend_ratio=0.25

​            )



​        \# 2) confusion-like

​        if os.path.exists(confusion_path):

​            confusion_pred = np.load(confusion_path)

​            meta_path = os.path.join("./models", "metadata.pkl")

​            if os.path.exists(meta_path):

​                stats_dict = joblib.load(meta_path)

​                oh_groups = stats_dict.get("onehot_groups", [])

​            else:

​                oh_groups = []

​            if len(oh_groups) >= 2:

​                grpA = oh_groups[0]

​                grpB = oh_groups[1]

​                xcol_path = os.path.join("./models", mtype, "x_col_names.npy")

​                if os.path.exists(xcol_path):

​                    xcols = list(np.load(xcol_path, allow_pickle=True))

​                else:

​                    xcols = None

​                if xcols:

​                    row_labels = [xcols[cid] for cid in grpA]

​                    col_labels = [xcols[cid] for cid in grpB]

​                else:

​                    row_labels = [f"Class {i+1}" for i in range(len(grpA))]

​                    col_labels = [f"Class {i+1}" for i in range(len(grpB))]



​                ycol_path = os.path.join("./models", mtype, "y_col_names.npy")

​                if os.path.exists(ycol_path):

​                    y_cols = list(np.load(ycol_path, allow_pickle=True))

​                else:

​                    y_cols = None



​                out_conf = os.path.join(base_out, "confusion_matrix")

​                ensure_dir(out_conf)

​                plot_confusion_from_npy(

​                    confusion_pred,

​                    row_labels, col_labels,

​                    out_dir=out_conf,

​                    y_col_names=y_cols,

​                    cell_scale=1/3,

​                    colorbar_extend_ratio=0.25

​                )

​            else:

​                print("[WARN] not enough onehot groups => skip confusion matrix.")



​    print("\n[INFO] visualize_main => done.")



if __name__ == "__main__":

​    visualize_main()





我之前运行良好的可视化,也就是显示非常正常的代码如下,给你做一个参考:





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



\# 全局字体

matplotlib.rcParams['font.size'] = 13

matplotlib.rcParams['axes.labelsize'] = 13

matplotlib.rcParams['axes.labelweight'] = 'bold'

matplotlib.rcParams['axes.titlesize'] = 15

matplotlib.rcParams['axes.titleweight'] = 'bold'

matplotlib.rcParams['xtick.labelsize'] = 12

matplotlib.rcParams['ytick.labelsize'] = 12

matplotlib.rcParams['legend.fontsize'] = 12





def ensure_fig_dir(dir_path=FIG_DIR):

​    """

​    Ensure that the directory 'dir_path' exists.

​    If not provided, default is FIG_DIR = './evaluation/figures'.

​    """

​    os.makedirs(dir_path, exist_ok=True)

​    return dir_path



\# ------------------------------

\# 对数值-数值, 类别-类别, 数值-类别的相关性

\# ------------------------------

def cramers_v(x, y):

​    confusion_matrix = pd.crosstab(x, y)

​    chi2 = ss.chi2_contingency(confusion_matrix)[0]

​    n = confusion_matrix.sum().sum()

​    r, k = confusion_matrix.shape

​    phi2 = max(0, chi2 - (k - 1)*(r - 1)/(n-1))

​    r_adj = r - (r-1)**2/(n-1)

​    k_adj = k - (k-1)**2/(n-1)

​    denom = min(k_adj-1, r_adj-1)

​    if denom == 0:

​        return 0.0

​    else:

​        return math.sqrt(phi2/denom)



def correlation_ratio(categories, values):

​    df = pd.DataFrame({'cat': categories, 'val': values})

​    df.dropna(inplace=True)

​    group_means = df.groupby('cat')['val'].mean()

​    mean_all = df['val'].mean()



​    ss_between = 0

​    for cat_value, group_mean in group_means.items():

​        group = df[df['cat'] == cat_value]

​        n = len(group)

​        ss_between += n * (group_mean - mean_all) ** 2



​    ss_total = ((df['val'] - mean_all) ** 2).sum()

​    if ss_total == 0:

​        return 0.0

​    eta = math.sqrt(ss_between / ss_total)

​    return eta





def mixed_correlation_matrix(X, col_names, numeric_cols_idx, method_numeric="pearson", dropna=True):

​    """

​    计算混合变量相关性矩阵（数值-数值, 类别-类别, 数值-类别）。

​    X: numpy.ndarray 或者 pandas.DataFrame, shape = (n_samples, n_features)

​    col_names: 每列的名字(list)

​    numeric_cols_idx: 哪些列是数值列 (list of int)

​    """

​    n_features = X.shape[1]

​    if not isinstance(X, pd.DataFrame):

​        X = pd.DataFrame(X, columns=col_names)



​    numeric_cols_set = set(numeric_cols_idx)

​    corr_matrix = np.zeros((n_features, n_features), dtype=float)

​    used_method_matrix = [["" for _ in range(n_features)] for _ in range(n_features)]



​    for i in range(n_features):

​        for j in range(n_features):

​            if i == j:

​                corr_matrix[i, j] = 1.0

​                used_method_matrix[i][j] = "self"

​                continue

​            if i > j:

​                corr_matrix[i, j] = corr_matrix[j, i]

​                used_method_matrix[i][j] = used_method_matrix[j][i]

​                continue



​            col_i = col_names[i]

​            col_j = col_names[j]

​            data_i = X[col_i]

​            data_j = X[col_j]

​            if dropna:

​                valid_mask = ~data_i.isna() & ~data_j.isna()

​                data_i = data_i[valid_mask]

​                data_j = data_j[valid_mask]



​            i_is_num = (i in numeric_cols_set)

​            j_is_num = (j in numeric_cols_set)



​            if i_is_num and j_is_num:

​                if method_numeric.lower() == "pearson":

​                    r, _ = ss.pearsonr(data_i, data_j)

​                    used_method = "pearson"

​                elif method_numeric.lower() == "spearman":

​                    r, _ = ss.spearmanr(data_i, data_j)

​                    used_method = "spearman"

​                elif method_numeric.lower() == "kendall":

​                    r, _ = ss.kendalltau(data_i, data_j)

​                    used_method = "kendall"

​                else:

​                    raise ValueError(f"Unsupported numeric correlation method: {method_numeric}")

​                corr_matrix[i, j] = r

​                used_method_matrix[i][j] = used_method

​            elif (not i_is_num) and (not j_is_num):

​                \# 类别-类别

​                r = cramers_v(data_i, data_j)

​                corr_matrix[i, j] = r

​                used_method_matrix[i][j] = "cramers_v"

​            else:

​                \# 数值-类别

​                if i_is_num:

​                    cat_data, num_data = data_j, data_i

​                else:

​                    cat_data, num_data = data_i, data_j

​                r = correlation_ratio(cat_data, num_data)

​                corr_matrix[i, j] = r

​                used_method_matrix[i][j] = "corr_ratio"



​    for i in range(n_features):

​        for j in range(i):

​            corr_matrix[i, j] = corr_matrix[j, i]

​            used_method_matrix[i][j] = used_method_matrix[j][i]



​    return corr_matrix, used_method_matrix





def plot_correlation_heatmap(X,

​                             col_names,

​                             numeric_cols_idx,

​                             filename="correlation_heatmap.jpg",

​                             method_numeric="pearson",

​                             cmap="ocean",

​                             vmin=-1,

​                             vmax=1):

​    """

​    绘制混合变量相关性热力图，并保存。

​    """

​    ensure_fig_dir()

​    corr_matrix, used_methods = mixed_correlation_matrix(

​        X, col_names, numeric_cols_idx,

​        method_numeric=method_numeric

​    )



​    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(col_names)), max(8, 0.5 * len(col_names))))



​    sns.heatmap(corr_matrix,

​                xticklabels=col_names,

​                yticklabels=col_names,

​                cmap=cmap,

​                annot=False,

​                square=True,

​                vmin=vmin,

​                vmax=vmax,

​                ax=ax,

​                cbar_kws={"shrink": 0.8, "aspect": 30, "label": "Correlation"})



​    ax.set_title("Mixed Correlation Heatmap", fontsize=14)

​    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)

​    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=12)



​    plt.tight_layout()

​    save_path = os.path.join(FIG_DIR, filename)

​    plt.savefig(save_path, dpi=700)

​    plt.close()

​    print(f"[plot_correlation_heatmap] Saved => {save_path}")





def plot_loss_curve(train_losses, val_losses, filename="loss_curve.jpg"):

​    """

​    Plot the training and validation loss vs. epochs.

​    """

​    ensure_fig_dir()

​    plt.figure()

​    plt.plot(train_losses, label='Train Loss', linewidth=2)

​    plt.plot(val_losses, label='Val Loss', linewidth=2)

​    plt.xlabel("Epoch")

​    plt.ylabel("Loss")

​    plt.legend()

​    plt.title("Training/Validation Loss")

​    save_path = os.path.join(FIG_DIR, filename)

​    plt.savefig(save_path, dpi=700, format='jpg')

​    plt.close()





def plot_scatter_3d_outputs_mse(y_true, y_pred, y_labels=None, filename="scatter_3d_output_mse.jpg"):

​    """

​    Create subplots for multi-output regression (dimension = n_outputs).

​    Each subplot:

​      \- X轴: True Y

​      \- Y轴: Predicted Y

​      \- Color: MSE (squared error) per sample for that output dimension.

​      \- 在标题处显示该维度的 R² 值.

​    """

​    ensure_fig_dir()

​    if y_pred.ndim != 2:

​        raise ValueError(f"y_pred shape expected (N, n_outputs), got {y_pred.shape}")

​    _, n_outputs = y_pred.shape



​    fig, axes = plt.subplots(1, n_outputs, figsize=(4 * n_outputs, 4), squeeze=False)



​    for i in range(n_outputs):

​        errors = (y_true[:, i] - y_pred[:, i]) ** 2

​        r2_val = r2_score(y_true[:, i], y_pred[:, i])



​        ax = axes[0, i]

​        sc = ax.scatter(

​            y_true[:, i],

​            y_pred[:, i],

​            c=errors,

​            alpha=0.5,

​            cmap='brg'

​        )

​        min_val = min(y_true[:, i].min(), y_pred[:, i].min())

​        max_val = max(y_true[:, i].max(), y_pred[:, i].max())

​        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)



​        if y_labels and i < len(y_labels):

​            title_str = f"{y_labels[i]} (MSE colormap)\nR² = {r2_val:.3f}"

​            ax.set_xlabel(f"True {y_labels[i]}")

​            ax.set_ylabel(f"Pred {y_labels[i]}")

​        else:

​            title_str = f"Output {i} (MSE colormap)\nR² = {r2_val:.3f}"

​            ax.set_xlabel("True")

​            ax.set_ylabel("Pred")



​        ax.set_title(title_str)



​        cbar = fig.colorbar(sc, ax=ax)

​        cbar.set_label("Squared Error", fontweight='bold', fontsize=13)

​        cbar.ax.tick_params(labelsize=12)



​    plt.tight_layout()

​    save_path = os.path.join(FIG_DIR, filename)

​    plt.savefig(save_path, dpi=700, format='jpg')

​    plt.close()





def plot_scatter_3d_outputs_mae(y_true, y_pred, y_labels=None, filename="scatter_3d_output_mae.jpg"):

​    ensure_fig_dir()

​    if y_pred.ndim != 2:

​        raise ValueError(f"y_pred shape expected (N, n_outputs), got {y_pred.shape}")

​    _, n_outputs = y_pred.shape



​    fig, axes = plt.subplots(1, n_outputs, figsize=(4 * n_outputs, 4), squeeze=False)



​    for i in range(n_outputs):

​        errors = np.abs(y_true[:, i] - y_pred[:, i])

​        r2_val = r2_score(y_true[:, i], y_pred[:, i])



​        ax = axes[0, i]

​        sc = ax.scatter(

​            y_true[:, i],

​            y_pred[:, i],

​            c=errors,

​            alpha=0.5,

​            cmap='ocean'

​        )

​        min_val = min(y_true[:, i].min(), y_pred[:, i].min())

​        max_val = max(y_true[:, i].max(), y_pred[:, i].max())

​        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)



​        if y_labels and i < len(y_labels):

​            title_str = f"{y_labels[i]} (MAE colormap)\nR² = {r2_val:.3f}"

​            ax.set_xlabel(f"True {y_labels[i]}")

​            ax.set_ylabel(f"Pred {y_labels[i]}")

​        else:

​            title_str = f"Output {i} (MAE colormap)\nR² = {r2_val:.3f}"

​            ax.set_xlabel("True")

​            ax.set_ylabel("Pred")



​        ax.set_title(title_str)



​        cbar = fig.colorbar(sc, ax=ax)

​        cbar.set_label("Absolute Error", fontweight='bold', fontsize=13)

​        cbar.ax.tick_params(labelsize=12)



​    plt.tight_layout()

​    save_path = os.path.join(FIG_DIR, filename)

​    plt.savefig(save_path, dpi=700, format='jpg')

​    plt.close()





def plot_residual_histogram(

​        y_true, y_pred, y_labels=None,

​        cmap_name="coolwarm",

​        vmin=-45, vmax=45,

​        filename="residual_hist_bottom.jpg"

):

​    ensure_fig_dir()

​    residuals = y_true - y_pred

​    n_outputs = residuals.shape[1]



​    fig, axes = plt.subplots(1, n_outputs, figsize=(4 * n_outputs, 4.5))

​    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.88, wspace=0.3)



​    num_bins = 30

​    bins_array = np.linspace(vmin, vmax, num_bins + 1)



​    cmap = cm.get_cmap(cmap_name)

​    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)



​    for i in range(n_outputs):

​        ax = axes[i] if n_outputs > 1 else axes

​        hist_data, bin_edges, patches = ax.hist(

​            residuals[:, i],

​            bins=bins_array,

​            alpha=0.9,

​            edgecolor='none'

​        )

​        for b_idx, patch in enumerate(patches):

​            bin_center = 0.5 * (bin_edges[b_idx] + bin_edges[b_idx + 1])

​            patch.set_facecolor(cmap(norm(bin_center)))



​        if y_labels and i < len(y_labels):

​            ax.set_title(f"Residuals of {y_labels[i]}")

​        else:

​            ax.set_title(f"Output {i} Residual")



​        ax.set_xlabel("Residual")

​        ax.set_ylabel("Count")

​        ax.set_xlim(vmin, vmax)



​    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

​    sm.set_array([])

​    cbar = fig.colorbar(

​        sm,

​        ax=axes.ravel().tolist(),

​        orientation="horizontal",

​        fraction=0.07,

​        pad=0.20,

​        shrink=0.9

​    )

​    cbar.set_label("Residual Value", fontweight='bold', fontsize=13)

​    cbar.ax.tick_params(labelsize=12)



​    save_path = os.path.join(FIG_DIR, filename)

​    plt.savefig(save_path, dpi=300)

​    plt.close()

​    print(f"[plot_residual_histogram] => {save_path}")





def plot_residual_kde(

​    y_true, y_pred, y_labels=None,

​    cmap_name="coolwarm",

​    vmin=-45, vmax=45,

​    filename="residual_kde_bottom.jpg"

):

​    ensure_fig_dir()

​    residuals = y_true - y_pred

​    n_outputs = residuals.shape[1]



​    fig, axes = plt.subplots(1, n_outputs, figsize=(4*n_outputs, 4.5))

​    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.88, wspace=0.3)



​    cmap = cm.get_cmap(cmap_name)

​    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)



​    for i in range(n_outputs):

​        ax = axes[i] if n_outputs>1 else axes

​        sns.kdeplot(residuals[:, i], ax=ax, fill=False, color="black")

​        lines = ax.get_lines()

​        if not lines:

​            continue

​        line = lines[-1]

​        x_plot = line.get_xdata()

​        y_plot = line.get_ydata()



​        idxsort = np.argsort(x_plot)

​        x_plot = x_plot[idxsort]

​        y_plot = y_plot[idxsort]



​        for j in range(len(x_plot)-1):

​            x0, x1 = x_plot[j], x_plot[j+1]

​            y0, y1 = y_plot[j], y_plot[j+1]

​            xmid = 0.5*(x0 + x1)

​            color = cmap(norm(xmid))

​            verts = np.array([

​                [x0, 0],

​                [x0, y0],

​                [x1, y1],

​                [x1, 0]

​            ])

​            poly = PolyCollection([verts], facecolors=[color], edgecolor='none', alpha=0.6)

​            ax.add_collection(poly)



​        if y_labels and i < len(y_labels):

​            ax.set_title(f"Residual KDE of {y_labels[i]}")

​        else:

​            ax.set_title(f"KDE - Output {i}")



​        ax.set_xlabel("Residual")

​        ax.set_ylabel("Density")

​        ax.set_xlim(vmin, vmax)



​    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

​    sm.set_array([])

​    cbar = fig.colorbar(

​        sm, ax=axes.ravel().tolist(),

​        orientation="horizontal",

​        fraction=0.07,

​        pad=0.20,

​        shrink=0.9

​    )

​    cbar.set_label("Residual Value", fontweight='bold', fontsize=13)

​    cbar.ax.tick_params(labelsize=12)



​    save_path = os.path.join(FIG_DIR, filename)

​    plt.savefig(save_path, dpi=300)

​    plt.close()

​    print(f"[plot_residual_kde] => {save_path}")





def plot_rf_feature_importance_bar(

​        rf_model,

​        feature_names,

​        filename="rf_feature_importance_bar.jpg",

​        top_k=20,

​        threshold=0.05

):

​    """

​    使用柱状图绘制随机森林(或类似)特征重要度

​    """

​    ensure_fig_dir()

​    importances = rf_model.feature_importances_

​    if importances is None or len(importances) == 0:

​        print("[WARNING] No importances found or importances is empty!")

​        return



​    sorted_indices = np.argsort(importances)[::-1]

​    topk_indices = sorted_indices[:top_k]



​    topk_features = [feature_names[i] for i in topk_indices]

​    topk_importances = importances[topk_indices]

​    colors = ["red" if imp > threshold else "blue" for imp in topk_importances]



​    fig, ax = plt.subplots(figsize=(8, 6))

​    bars = ax.barh(range(len(topk_importances)), topk_importances, align="center", color=colors)



​    ax.set_yticks(range(len(topk_importances)))

​    ax.set_yticklabels(topk_features, fontsize=10)

​    ax.invert_yaxis()



​    ax.set_xlabel("Feature Importance", fontsize=12)

​    ax.set_title(f"Feature Importance (Top-{top_k})", fontsize=14, fontweight='bold')



​    ax.axvspan(0, threshold, facecolor='lightgray', alpha=0.5)

​    ax.axvline(x=threshold, color='red', linestyle='dashed', linewidth=2)



​    legend_elements = [

​        Patch(facecolor="red", label=f"Importance > {threshold}"),

​        Patch(facecolor="royalblue", label=f"Importance ≤ {threshold}")

​    ]

​    ax.legend(handles=legend_elements, loc="lower right", fontsize=12)

​    plt.tight_layout()

​    save_path = os.path.join(FIG_DIR, filename)

​    plt.savefig(save_path, dpi=700)

​    plt.close()



​    print(f"[plot_rf_feature_importance_bar] Saved => {save_path}")





def plot_hbar_with_mean(

​    ax,

​    model_names,

​    values,

​    subplot_label="(a)",

​    metric_label="Metric",

​    bigger_is_better=False,

​    width=0.4

):

​    arr = np.array(values)

​    if bigger_is_better:

​        best_idx = arr.argmax()

​        worst_idx = arr.argmin()

​    else:

​        best_idx = arr.argmin()

​        worst_idx = arr.argmax()



​    colors = []

​    for i, val in enumerate(arr):

​        if i == best_idx:

​            colors.append("red")

​        elif i == worst_idx:

​            colors.append("blue")

​        else:

​            colors.append("green")



​    mean_val = arr.mean()

​    y_positions = np.arange(len(arr))

​    ax.barh(y_positions, arr, color=colors, alpha=0.8, height=width)

​    ax.set_yticks(y_positions)

​    ax.set_yticklabels(model_names)

​    ax.invert_yaxis()

​    ax.text(-0.08, 1.05, subplot_label, transform=ax.transAxes,

​            ha="left", va="top", fontsize=14, fontweight="bold")

​    ax.set_title(metric_label, fontsize=14, fontweight='bold')



​    for i, v in enumerate(arr):

​        ax.text(v, i, f"{v:.2f}", ha=("left" if v>=0 else "right"),

​                va="center", fontsize=10)



​    ax.axvline(mean_val, color='gray', linestyle='--', linewidth=2)

​    xmin, xmax = sorted([0, mean_val])

​    ax.axvspan(xmin, xmax, facecolor='gray', alpha=0.2)



​    max_val = arr.max()

​    min_val = arr.min()

​    if min_val < 0:

​        ax.set_xlim(min_val * 1.1, max_val * 1.79)

​    else:

​        ax.set_xlim(0, max_val * 1.79)



​    legend_elements = [

​        Patch(facecolor="red", label="Best"),

​        Patch(facecolor="blue", label="Worst"),

​        Patch(facecolor="green", label="Ordinary"),

​        Patch(facecolor="gray", alpha=0.2, label="Under Mean"),

​    ]

​    ax.legend(handles=legend_elements, loc="lower right")





def plot_three_metrics_horizontal(

​    metrics_data,

​    save_name="three_metrics_horizontal.jpg"

):

​    ensure_fig_dir()

​    model_names = list(metrics_data.keys())

​    mse_vals = [metrics_data[m]["MSE"] for m in model_names]

​    mae_vals = [metrics_data[m]["MAE"] for m in model_names]

​    r2_vals  = [metrics_data[m]["R2"]  for m in model_names]



​    fig, axes = plt.subplots(1, 3, figsize=(16, 5))



​    plot_hbar_with_mean(

​        ax=axes[0],

​        model_names=model_names,

​        values=mse_vals,

​        subplot_label="(a)",

​        metric_label="MSE (Lower is better)",

​        bigger_is_better=False,

​        width=0.4

​    )

​    plot_hbar_with_mean(

​        ax=axes[1],

​        model_names=model_names,

​        values=mae_vals,

​        subplot_label="(b)",

​        metric_label="MAE (Lower is better)",

​        bigger_is_better=False,

​        width=0.4

​    )

​    plot_hbar_with_mean(

​        ax=axes[2],

​        model_names=model_names,

​        values=r2_vals,

​        subplot_label="(c)",

​        metric_label="R2 (Higher is better)",

​        bigger_is_better=True,

​        width=0.4

​    )

​    plt.tight_layout()

​    out_path = os.path.join(FIG_DIR, save_name)

​    plt.savefig(out_path, dpi=300)

​    plt.close()

​    print(f"[plot_three_metrics_horizontal] => {out_path}")





from matplotlib.patches import Patch



def plot_hbar_with_threshold(

​        ax,

​        model_names,

​        values,

​        subplot_label="(a)",

​        metric_label="Metric",

​        bigger_is_better=False,

​        width=0.4,

​        threshold_h=0.5,

​        threshold_l=0.0

):

​    arr = np.array(values)

​    if bigger_is_better:

​        best_idx = arr.argmax()

​        worst_idx = arr.argmin()

​    else:

​        best_idx = arr.argmin()

​        worst_idx = arr.argmax()



​    colors = []

​    for i, val in enumerate(arr):

​        if i == best_idx:

​            colors.append("red")

​        elif i == worst_idx:

​            colors.append("blue")

​        else:

​            colors.append("green")



​    y_positions = np.arange(len(arr))

​    ax.barh(y_positions, arr, color=colors, alpha=0.8, height=width)

​    ax.set_yticks(y_positions)

​    ax.set_yticklabels(model_names)

​    ax.invert_yaxis()

​    ax.text(-0.08, 1.05, subplot_label, transform=ax.transAxes,

​            ha="left", va="top", fontsize=14, fontweight="bold")

​    ax.set_title(metric_label, fontsize=14, fontweight='bold')



​    for i, v in enumerate(arr):

​        ax.text(v, i, f"{v:.2f}", ha=("left" if v >= 0 else "right"),

​                va="center", fontsize=10)



​    if threshold_l == 0.0:

​        ax.axvspan(threshold_l, threshold_h, facecolor='gray', alpha=0.2)

​        legend_elements = [

​            Patch(facecolor="gray", alpha=0.2, label="Acceptable")

​        ]

​    else:

​        ax.axvspan(0, threshold_l, facecolor='gray', alpha=0.2)

​        ax.axvspan(threshold_l, threshold_h, facecolor='lightcoral', alpha=0.3)

​        ax.axvline(threshold_l, color='gray', linestyle='--', linewidth=2)

​        ax.axvline(threshold_h, color='gray', linestyle='--', linewidth=2)

​        legend_elements = [

​            Patch(facecolor="gray", alpha=0.2, label="Acceptable"),

​            Patch(facecolor="lightcoral", alpha=0.3, label="Overfitting Risk")

​        ]



​    max_val = arr.max()

​    min_val = arr.min()

​    if min_val < 0:

​        ax.set_xlim(min_val * 1.1, max_val * 1.79)

​    else:

​        ax.set_xlim(0, max_val * 1.79)



​    legend_elements.extend([

​        Patch(facecolor="red", label="Best"),

​        Patch(facecolor="blue", label="Worst"),

​        Patch(facecolor="green", label="Ordinary")

​    ])

​    ax.legend(handles=legend_elements, loc="lower right")





def plot_overfitting_horizontal(

​    overfit_data,

​    save_name="overfitting_horizontal.jpg"

):

​    ensure_fig_dir()

​    model_names = list(overfit_data.keys())



​    msr_vals = [overfit_data[m]["MSE_ratio"] for m in model_names]

​    r2d_vals = [overfit_data[m]["R2_diff"]   for m in model_names]



​    fig, axes = plt.subplots(1, 2, figsize=(12, 5))



​    plot_hbar_with_threshold(

​        ax=axes[0],

​        model_names=model_names,

​        values=msr_vals,

​        subplot_label="(a)",

​        metric_label="MSE Ratio (Val/Train)\n(Lower is better)",

​        bigger_is_better=False,

​        width=0.4,

​        threshold_h=10

​    )

​    plot_hbar_with_threshold(

​        ax=axes[1],

​        model_names=model_names,

​        values=r2d_vals,

​        subplot_label="(b)",

​        metric_label="R2 diff (Train - Val)\n(Lower is better)",

​        bigger_is_better=False,

​        width=0.4,

​        threshold_h=0.2,

​        threshold_l=0.15

​    )



​    plt.tight_layout()

​    out_path = os.path.join(FIG_DIR, save_name)

​    plt.savefig(out_path, dpi=300)

​    plt.close()

​    print(f"[plot_overfitting_horizontal] => {out_path}")





\# ========== 数据分析可视化 ==========



def plot_kde_distribution(df, columns, filename="kde_distribution.jpg", out_dir="./evaluation/figures"):

​    """

​    绘制各变量的 KDE 分布

​    每个子图一个colorbar

​    """

​    fig_dir = ensure_fig_dir(out_dir)



​    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 5))

​    if len(columns) == 1:

​        axes = [axes]



​    for i, col in enumerate(columns):

​        ax = axes[i]

​        if col not in df.columns:

​            ax.text(0.5, 0.5, f"'{col}' not in df.columns", ha='center', va='center')

​            continue



​        \# sns.kdeplot(df[col], ax=ax, fill=False, color="black")

​        sns.kdeplot(df[col], ax=ax, fill=False, color="black", clip=(df[col].min(), df[col].max()))

​        lines = ax.get_lines()

​        if not lines:

​            ax.set_title(f"No Data for {col}")

​            continue



​        line = lines[-1]

​        x_plot, y_plot = line.get_xdata(), line.get_ydata()

​        idxsort = np.argsort(x_plot)

​        x_plot, y_plot = x_plot[idxsort], y_plot[idxsort]



​        vmin = max(np.min(x_plot), df[col].min())   #发现负数

​        vmax = min(np.max(x_plot), df[col].max())

​        cmap = cm.get_cmap("coolwarm")

​        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)



​        for j in range(len(x_plot)-1):

​            x0, x1 = x_plot[j], x_plot[j+1]

​            y0, y1 = y_plot[j], y_plot[j+1]

​            color = cmap(norm((x0 + x1)*0.5))

​            verts = np.array([

​                [x0, 0],

​                [x0, y0],

​                [x1, y1],

​                [x1, 0]

​            ])

​            poly = PolyCollection([verts], facecolors=[color], edgecolor='none', alpha=0.6)

​            ax.add_collection(poly)



​        ax.set_title(f"KDE of {col}")

​        ax.set_xlabel(col)

​        ax.set_ylabel("Density")

​        ax.set_xlim(df[col].min(), df[col].max())



​        sm = cm.ScalarMappable(norm=norm, cmap=cmap)

​        sm.set_array([])

​        cb = plt.colorbar(sm, ax=ax)

​        cb.set_label("Value Range", fontweight='bold', fontsize=12)

​        cb.ax.tick_params(labelsize=10)



​    plt.tight_layout()

​    save_path = os.path.join(fig_dir, filename)

​    plt.savefig(save_path, dpi=300)

​    plt.close()

​    print(f"[plot_kde_distribution] => {save_path}")





def plot_catalyst_size_vs_product(df, filename="catalyst_size_vs_product.jpg", out_dir="./evaluation/figures"):

​    """

​    绘制催化剂尺寸 vs 产物产量（散点图）

​    x='Catalyst size', y=product, hue='Active metal'

​    """

​    fig_dir = ensure_fig_dir(out_dir)



​    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

​    products = ['H2', 'CO', 'C1', 'C2+']



​    for i, product in enumerate(products):

​        ax = axes[i // 2, i % 2]

​        needed_cols = ['Particle size (nm)', 'Active metal', product]

​        if all(col in df.columns for col in needed_cols):

​            sns.scatterplot(x='Particle size (nm)', y=product, hue='Active metal', data=df, ax=ax, alpha=0.7)

​            ax.set_title(f'Particle size (nm) vs {product} Yield')

​            ax.set_xlabel('Particle size (nm)')  # or mm if needed

​            ax.set_ylabel(f'{product} Yield (%)')

​        else:

​            ax.text(0.5, 0.5, f"Cols not found for {product}", ha='center', va='center')

​    plt.tight_layout()

​    save_path = os.path.join(fig_dir, filename)

​    plt.savefig(save_path, dpi=300)

​    plt.close()

​    print(f"[plot_catalyst_size_vs_product] => {save_path}")





def plot_potential_vs_product_by_electrolyte(df, filename="potential_vs_product_by_electrolyte.jpg",

​                                             out_dir="./evaluation/figures"):

​    """

​    绘制【电解质】分组下的电位 vs 产物散点, hue='Electrode support' 仅作为示例.

​    """

​    fig_dir = ensure_fig_dir(out_dir)



​    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

​    products = ['H2', 'CO', 'C1', 'C2+']



​    for i, product in enumerate(products):

​        ax = axes[i // 2, i % 2]

​        needed_cols = ['Potential (V vs. RHE)', 'Electrode support', product]

​        if all(col in df.columns for col in needed_cols):

​            sns.scatterplot(x='Potential (V vs. RHE)', y=product, hue='Electrode support',

​                            data=df, ax=ax, alpha=0.7)

​            ax.set_title(f'Potential vs {product} (Electrode support Hue)')

​            ax.set_xlabel('Potential (V vs. RHE)')

​            ax.set_ylabel(f'{product} Yield (%)')

​        else:

​            ax.text(0.5,0.5,f"Cols not found for {product}", ha='center', va='center')

​    plt.tight_layout()

​    save_path = os.path.join(fig_dir, filename)

​    plt.savefig(save_path, dpi=300)

​    plt.close()

​    print(f"[plot_potential_vs_product_by_electrolyte] => {save_path}")





def plot_product_distribution_by_catalyst_and_potential(df, filename="product_distribution.jpg",

​                                                       out_dir="./evaluation/figures"):

​    """

​    绘制不同催化剂和电位下产物分布的盒须图

​    需求列: 'Active metal','Potential','H2','CO','C1','C2+'

​    """

​    fig_dir = ensure_fig_dir(out_dir)



​    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

​    products = ['H2', 'CO', 'C1', 'C2+']



​    if 'Potential (V vs. RHE)' in df.columns:

​        df['Potential_bin'] = pd.cut(df['Potential (V vs. RHE)'], bins=5)

​    else:

​        df['Potential_bin'] = "Unknown"



​    for i, product in enumerate(products):

​        ax = axes[i]

​        needed_cols = ['Active metal', product, 'Potential_bin']

​        if all(col in df.columns for col in needed_cols):

​            sns.boxplot(x='Active metal', y=product, hue='Potential_bin', data=df, ax=ax)

​            ax.set_title(f'{product} by Active Metal & Potential')

​            ax.tick_params(axis='x', rotation=45)

​        else:

​            ax.text(0.5,0.5,f"Cols not found for {product}", ha='center', va='center')

​    plt.tight_layout()

​    save_path = os.path.join(fig_dir, filename)

​    plt.savefig(save_path, dpi=300)

​    plt.close()

​    print(f"[plot_product_distribution_by_catalyst_and_potential] => {save_path}")





\# ============【新增】三张boxplot + 一张 potential vs product ============



def plot_product_vs_potential_bin(df, filename="box_product_vs_potential_bin.jpg", out_dir="./evaluation/figures"):

​    """

​    (1) 产物 vs Potential_bin 盒须图

​    需求列: "Potential","H2","CO","C1","C2+"

​    """

​    fig_dir = ensure_fig_dir(out_dir)

​    products = ["H2","CO","C1","C2+"]



​    if 'Potential (V vs. RHE)' not in df.columns:

​        print("[WARN] 'Potential (V vs. RHE)' not in df => skip boxplot")

​        return



​    \# bin

​    df['Potential_bin_custom'] = pd.cut(df['Potential (V vs. RHE)'], bins=5)

​    fig, axes = plt.subplots(1, 4, figsize=(20,5))

​    for i, product in enumerate(products):

​        ax = axes[i]

​        if product not in df.columns:

​            ax.text(0.5,0.5,f"'{product}' not in df", ha='center', va='center')

​            continue

​        sns.boxplot(x='Potential_bin_custom', y=product, data=df, ax=ax)

​        ax.set_title(f"{product} vs Potential Bin")

​        ax.tick_params(axis='x', rotation=45)

​    plt.tight_layout()

​    out_path = os.path.join(fig_dir, filename)

​    plt.savefig(out_path, dpi=300)

​    plt.close()

​    print(f"[plot_product_vs_potential_bin] => {out_path}")





def plot_product_vs_shape(df, filename="box_product_vs_shape.jpg", out_dir="./evaluation/figures"):

​    """

​    (2) 产物 vs Shape 盒须图

​    需求列: "Shape","H2","CO","C1","C2+"

​    """

​    fig_dir = ensure_fig_dir(out_dir)

​    products = ["H2","CO","C1","C2+"]



​    if 'Shape' not in df.columns:

​        print("[WARN] 'Shape' not in df => skip boxplot shape")

​        return



​    fig, axes = plt.subplots(1, 4, figsize=(20,5))

​    for i, product in enumerate(products):

​        ax = axes[i]

​        if product not in df.columns:

​            ax.text(0.5,0.5,f"'{product}' not in df", ha='center', va='center')

​            continue

​        sns.boxplot(x='Shape', y=product, data=df, ax=ax)

​        ax.set_title(f"{product} vs Shape")

​        ax.tick_params(axis='x', rotation=45)

​    plt.tight_layout()

​    out_path = os.path.join(fig_dir, filename)

​    plt.savefig(out_path, dpi=300)

​    plt.close()

​    print(f"[plot_product_vs_shape] => {out_path}")





def plot_product_vs_catalyst(df, filename="box_product_vs_catalyst.jpg", out_dir="./evaluation/figures"):

​    """

​    (3) 产物 vs Catalyst 盒须图

​    需求列: "Catalyst","H2","CO","C1","C2+"

​    """

​    fig_dir = ensure_fig_dir(out_dir)

​    products = ["H2","CO","C1","C2+"]



​    if 'Active metal' not in df.columns:

​        print("[WARN] 'Active metal' not in df => skip boxplot catalyst")

​        return



​    fig, axes = plt.subplots(1, 4, figsize=(20,5))

​    for i, product in enumerate(products):

​        ax = axes[i]

​        if product not in df.columns:

​            ax.text(0.5,0.5,f"'{product}' not in df", ha='center', va='center')

​            continue

​        sns.boxplot(x='Active metal', y=product, data=df, ax=ax)

​        ax.set_title(f"{product} vs Active metal")

​        ax.tick_params(axis='x', rotation=45)

​    plt.tight_layout()

​    out_path = os.path.join(fig_dir, filename)

​    plt.savefig(out_path, dpi=300)

​    plt.close()

​    print(f"[plot_product_vs_catalyst] => {out_path}")





def plot_potential_vs_product(df, filename="three_dot_potential_vs_product.jpg", out_dir="./evaluation/figures"):

​    """

​    (4) 仅考虑 Potential vs. Product (H2/CO/C1/C2+) 三个点, 不区分别的hue

​       这里随意三点 => 你可自定义

​    """

​    fig_dir = ensure_fig_dir(out_dir)

​    products = ["H2","CO","C1","C2+"]

​    if 'Potential (V vs. RHE)' not in df.columns:

​        print("[WARN] 'Potential (V vs. RHE)' not in df => skip potential vs product")

​        return



​    plt.figure(figsize=(7,6))

​    for product in products:

​        if product in df.columns:

​            plt.scatter(df['Potential (V vs. RHE)'], df[product], label=product, alpha=0.7)



​    plt.title("Potential vs Products (3 dot style)")

​    plt.xlabel("Potential (V vs. RHE)")

​    plt.ylabel("Yield (%)")

​    plt.legend()

​    out_path = os.path.join(fig_dir, filename)

​    plt.savefig(out_path, dpi=300)

​    plt.close()

​    print(f"[plot_potential_vs_product] => {out_path}")





\# ============ plot_advanced_data_analysis ============

def plot_advanced_data_analysis(df, out_dir):

​    """

​    调用若干函数做数据分析可视化(如KDE,散点,盒须图等).

​    针对你的列: 'Potential','catalyst_size','electrolyte','H2','CO','C1','C2+','Catalyst','Shape', ...

​    新增三张boxplot + 一个potential vs product散点

​    """

​    fig_dir = ensure_fig_dir(out_dir)



​    \# 1) KDE 分布

​    possible_cols = [c for c in ["Potential (V vs. RHE)","H2","CO","C1","C2+","Particle size (nm)"] if c in df.columns]

​    if len(possible_cols) > 0:

​        plot_kde_distribution(df, possible_cols, filename="kde_distribution.jpg", out_dir=fig_dir)



​    \# 2) catalyst_size vs product

​    plot_catalyst_size_vs_product(df, filename="catalyst_size_vs_product.jpg", out_dir=fig_dir)



​    \# 3) potential_vs_product_by_electrolyte

​    plot_potential_vs_product_by_electrolyte(df, filename="potential_vs_product_by_electrolyte.jpg", out_dir=fig_dir)



​    \# 4) product_distribution_by_catalyst_and_potential

​    plot_product_distribution_by_catalyst_and_potential(df, filename="product_distribution.jpg", out_dir=fig_dir)



​    \# ------------- 【新增】三张盒须图 + 一个潜势散点 -------------

​    \# (a) 产物 vs Potential_bin

​    plot_product_vs_potential_bin(df, filename="box_product_vs_potential_bin.jpg", out_dir=fig_dir)

​    \# (b) 产物 vs Shape

​    plot_product_vs_shape(df, filename="box_product_vs_shape.jpg", out_dir=fig_dir)

​    \# (c) 产物 vs Catalyst

​    plot_product_vs_catalyst(df, filename="box_product_vs_catalyst.jpg", out_dir=fig_dir)

​    \# (d) potential vs product 三点图

​    plot_potential_vs_product(df, filename="three_dot_potential_vs_product.jpg", out_dir=fig_dir)



​    print("[INFO] plot_advanced_data_analysis => done.")

\# --------------------------------------------------------



\# ========== 推理可视化(2D Heatmap + Confusion)保留 ==========



def model_predict(model, X_2d):

​    """

​    根据传入的模型类型(是否是PyTorch)进行预测

​    """

​    if hasattr(model, 'eval') and hasattr(model, 'forward'):

​        import torch

​        model.eval()

​        X_tensor = torch.tensor(X_2d, dtype=torch.float32)

​        with torch.no_grad():

​            out = model(X_tensor)

​        return out.cpu().numpy()

​    else:

​        \# sklearn / catboost / xgboost

​        return model.predict(X_2d)





def plot_2d_mimo_heatmaps(

​    grid_x, grid_y, predictions,

​    out_labels=None,

​    out_dir="./",

​    prefix="mimo_heatmap"

):

​    """

​    多输出回归结果的 2D 网格热力图 (一次性画 out_dim 张图).

​    predictions.shape = (H, W, out_dim)

​    """

​    H, W, out_dim = predictions.shape

​    if out_labels is None or len(out_labels)!=out_dim:

​        out_labels = [f"Output_{i+1}" for i in range(out_dim)]



​    fig_dir = ensure_fig_dir(out_dir)



​    for i in range(out_dim):

​        plt.figure(figsize=(6,5))

​        z = predictions[:,:,i]

​        cm_ = plt.pcolormesh(grid_x, grid_y, z, shading='auto', cmap='viridis')

​        cb = plt.colorbar(cm_)

​        cb.set_label(f"{out_labels[i]}", fontsize=12)



​        plt.xlabel("potential")

​        plt.ylabel("catalyst_size")

​        plt.title(f"Heatmap of {out_labels[i]}")

​        out_fn = os.path.join(fig_dir, f"{prefix}_{i+1}.jpg")

​        plt.savefig(out_fn, dpi=150, bbox_inches='tight')

​        plt.close()





def plot_inference_2d_heatmap_for_two_continuous_vars(

​        model, x_col_names, numeric_cols_idx, stats_dict,

​        conti_var_x="potential", conti_var_y="catalyst_size",

​        n_points=50,

​        out_dir="./",

​        y_col_names=None,

​        scaler_x=None,

​        scaler_y=None

):

​    """

​    (来自之前示例) ...

​    """

​    \# ... 保留你之前的实现即可

​    pass  # 省略示例





def plot_inference_confusion_matrix_for_two_categories(

​        model, x_col_names, numeric_cols_idx, stats_dict,

​        group1_index=0,

​        group2_index=1,

​        out_dir="./",

​        y_col_names=None,

​        scaler_x=None,

​        scaler_y=None

):

​    """

​    (来自之前示例) ...

​    """

​    \# ... 保留你之前的实现即可

​    pass  # 省略示例



请你仔细思考我的图片布局问题,以及我的correlation_heatmap_one_hot.jpg的参数label读取问题. 

还有另外一个问题就是我最后的推理可视化(2D Heatmap + Confusion)的colorbar范围:colorbar范围 => [max(0, min_val*(1-extend)), min(100, max_val*(1+extend))]

这个东西说的是我读取的原始数据的min 和max 是最开始读取的那个用于画kde的min和max.请帮我把这个一并修改
plot_confusion_from_npy和plot_2d_heatmap_from_npy
还有就是这两个图也显示的很糟糕,那几个colorbar相互遮挡了,而且matrix里面的字也太大了超出了我的格子.

注意,你给我的代码中经常出现下列问题:[plot_correlation_heatmap] Saved => ./evaluation/figures/Dataset_20250214_final_3/DataCorrelation/correlation_heatmap.jpg
Traceback (most recent call last):
  File "visualization.py", line 322, in <module>
    visualize_main()
  File "visualization.py", line 91, in visualize_main
    plot_kde_distribution(df_raw_14, existing_cols, filename=out_kde)
  File "/media/herryao/81ca6f19-78c8-470d-b5a1-5f35b4678058/work_dir/Document/skku/with_asif/AI_4_SCIENCE/src/case0/NewValPLTKFAI4CatalystProject/utils.py", line 550, in plot_kde_distribution
    plt.savefig(save_path, dpi=300)
  File "/home/herryao/anaconda3/envs/torch1.10/lib/python3.8/site-packages/matplotlib/pyplot.py", line 1023, in savefig
    res = fig.savefig(*args, **kwargs)
  File "/home/herryao/anaconda3/envs/torch1.10/lib/python3.8/site-packages/matplotlib/figure.py", line 3378, in savefig
    self.canvas.print_figure(fname, **kwargs)
  File "/home/herryao/anaconda3/envs/torch1.10/lib/python3.8/site-packages/matplotlib/backends/backend_qtagg.py", line 75, in print_figure
    super().print_figure(*args, **kwargs)
  File "/home/herryao/anaconda3/envs/torch1.10/lib/python3.8/site-packages/matplotlib/backend_bases.py", line 2366, in print_figure
    result = print_method(
  File "/home/herryao/anaconda3/envs/torch1.10/lib/python3.8/site-packages/matplotlib/backend_bases.py", line 2232, in <lambda>
    print_method = functools.wraps(meth)(lambda *args, **kwargs: meth(
  File "/home/herryao/anaconda3/envs/torch1.10/lib/python3.8/site-packages/matplotlib/backends/backend_agg.py", line 526, in print_jpg
    self._print_pil(filename_or_obj, "jpeg", pil_kwargs)
  File "/home/herryao/anaconda3/envs/torch1.10/lib/python3.8/site-packages/matplotlib/backends/backend_agg.py", line 458, in _print_pil
    mpl.image.imsave(
  File "/home/herryao/anaconda3/envs/torch1.10/lib/python3.8/site-packages/matplotlib/image.py", line 1689, in imsave
    image.save(fname, **pil_kwargs)
  File "/home/herryao/anaconda3/envs/torch1.10/lib/python3.8/site-packages/PIL/Image.py", line 2436, in save
    fp = builtins.open(filename, "w+b")
FileNotFoundError: [Errno 2] No such file or directory: './evaluation/figures/./evaluation/figures/Dataset_20250214_final_3/DataCorrelation/kde_distribution.jpg'

░▒▓      /media/h/8/w/Doc/s/w/A/src/case0/NewValPLTKFAI4CatalystProject  on    master +10 !62 ?16             took 5s    torch1.10    at 19:18:44   ▓▒░
❯ python visualization.py
请务必消除相关问题

请您帮我思考一下如何修改以上问题,给我完整的代码
