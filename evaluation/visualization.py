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


def ensure_fig_dir():
    """
    Ensure that the figure output directory exists.
    """
    os.makedirs(FIG_DIR, exist_ok=True)

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
    """
    ensure_fig_dir()

    if y_pred.ndim != 2:
        raise ValueError(f"y_pred shape expected (N, n_outputs), got {y_pred.shape}")
    n_samples, n_outputs = y_pred.shape

    fig, axes = plt.subplots(1, n_outputs, figsize=(4*n_outputs, 4), squeeze=False)

    for i in range(n_outputs):
        errors = (y_true[:, i] - y_pred[:, i])**2
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

        if y_labels and i < len(y_labels):
            ax.set_xlabel(f"True {y_labels[i]}")
            ax.set_ylabel(f"Pred {y_labels[i]}")
            ax.set_title(f"{y_labels[i]} (MSE colormap)")
        else:
            ax.set_xlabel("True")
            ax.set_ylabel("Pred")
            ax.set_title(f"Output {i} (MSE colormap)")

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
    """
    ensure_fig_dir()

    if y_pred.ndim != 2:
        raise ValueError(f"y_pred shape expected (N, n_outputs), got {y_pred.shape}")
    n_samples, n_outputs = y_pred.shape

    fig, axes = plt.subplots(1, n_outputs, figsize=(4*n_outputs, 4), squeeze=False)

    for i in range(n_outputs):
        errors = np.abs(y_true[:, i] - y_pred[:, i])  # MAE
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

        if y_labels and i < len(y_labels):
            ax.set_xlabel(f"True {y_labels[i]}")
            ax.set_ylabel(f"Pred {y_labels[i]}")
            ax.set_title(f"{y_labels[i]} (MAE colormap)")
        else:
            ax.set_xlabel("True")
            ax.set_ylabel("Pred")
            ax.set_title(f"Output {i} (MAE colormap)")

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