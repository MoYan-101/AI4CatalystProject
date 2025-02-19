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
import phik
from phik import phik_matrix
from mpl_toolkits.mplot3d import Axes3D

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def ensure_dir_for_file(filepath):
    dir_ = os.path.dirname(filepath)
    if dir_:
        os.makedirs(dir_, exist_ok=True)

def normalize_data(data, vmin, vmax):
    """归一化数据到 [0,1] 范围"""
    return (data - vmin) / (vmax - vmin) if vmax > vmin else data

# --------------- correlation ---------------
def short_label(s: str) -> str:
    """
    根据下划线将字符串分割，取最后一段，并做以下处理：
      - 若最后一段全是大写 (如 "CO", "OH")，则原样返回
      - 特定化学符号进行特例转换（如 "cu(oh)2" -> "Cu(OH)2"）
      - 其余情况：首字母大写，其他部分保持原状
    """
    special_chemicals = {
        "cu": "Cu",
        "cu(oh)2": "Cu(OH)2",
        "cuxo": "CuxO",
        "cu2s": "Cu2S",
        "cu2(oh)2co3": "Cu2(OH)2CO3"
    }

    parts = s.split('_')
    last_part = parts[-1]  # 取最后一段

    # 若最后一段是空字符串
    if not last_part:
        return last_part  # 返回空串即可

    # 先检查是否属于特例化学符号
    lower_last_part = last_part.lower()  # 转小写匹配
    if lower_last_part in special_chemicals:
        return special_chemicals[lower_last_part]

    # 如果最后一段全是大写 (含数字/符号不影响 isupper，只要字母全大写即可)
    if last_part.isupper():
        return last_part

    # 否则，仅将首字母转大写，其余部分保持原状
    return last_part[0].upper() + last_part[1:]


# --------------- 相关性可视化phik ---------------
def plot_phik_correlation_heatmap(df, filename,
                                  interval_cols=None,
                                  cmap="ocean",
                                  vmin=0, vmax=1):
    """
    使用 Phik 计算 df 各列间的相关系数 (0..1)，再画热力图。
    - df: pandas.DataFrame (带原始列名)
    - interval_cols: 哪些列是数值，需要自动分箱
    - 绘图时, 行列名都改成 short_label(...) 后的名字
    """
    ensure_dir_for_file(filename)

    if interval_cols is None:
        interval_cols = []

    # 1) 用原始 df 的列名来计算 phik
    phik_mat = df.phik_matrix(interval_cols=interval_cols)

    # 2) 计算完成后, 把 phik_mat 的行列都做 short_label
    old_cols = list(phik_mat.columns)
    new_cols = [short_label(c) for c in old_cols]

    # 改一下 phik_mat 的行列
    phik_mat2 = phik_mat.copy()
    phik_mat2.columns = new_cols
    phik_mat2.index = new_cols

    # 3) 画热力图
    fig, ax = plt.subplots(figsize=(max(10, 0.5*len(new_cols)), max(8, 0.5*len(new_cols))))
    sns.heatmap(phik_mat2,
                xticklabels=new_cols,
                yticklabels=new_cols,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                square=True,
                annot=True,
                fmt=".2f",
                cbar_kws={"shrink": 0.8, "aspect": 30, "label": "Phik Correlation"},
                ax=ax)
    ax.set_title("Phik Correlation Heatmap", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()
    print(f"[plot_phik_correlation_heatmap] => {filename}")

# --------------- 训练可视化: Loss, scatter, residual, etc. ---------------
def plot_loss_curve(train_losses, val_losses, filename):
    ensure_dir_for_file(filename)
    plt.figure()
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training/Validation Loss")
    plt.savefig(filename, dpi=700, format='jpg')
    plt.close()

def plot_scatter_3d_outputs_mse(y_true, y_pred, y_labels=None, filename="scatter_3d_mse.jpg"):
    ensure_dir_for_file(filename)
    if y_pred.ndim!=2:
        raise ValueError("y_pred must be 2D (N, out_dim)")
    _, out_dim= y_pred.shape
    fig, axes= plt.subplots(1,out_dim, figsize=(4*out_dim,4), squeeze=False)
    for i in range(out_dim):
        errors= (y_true[:,i]-y_pred[:,i])**2
        r2_val= r2_score(y_true[:,i], y_pred[:,i])
        ax= axes[0,i]
        sc= ax.scatter(y_true[:,i], y_pred[:,i], c=errors, alpha=0.5, cmap='brg')
        min_val= min(y_true[:,i].min(), y_pred[:,i].min())
        max_val= max(y_true[:,i].max(), y_pred[:,i].max())
        ax.plot([min_val, max_val],[min_val, max_val],'r--', linewidth=1.5)

        if y_labels and i<len(y_labels):
            ax.set_title(f"{y_labels[i]} (MSE)\nR²={r2_val:.3f}")
            ax.set_xlabel(f"True {y_labels[i]}")
            ax.set_ylabel(f"Pred {y_labels[i]}")
        else:
            ax.set_title(f"Out {i} (MSE)\nR²={r2_val:.3f}")

        cbar= fig.colorbar(sc, ax=ax)
        cbar.set_label("Squared Error")
    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()

def plot_scatter_3d_outputs_mae(y_true, y_pred, y_labels=None, filename="scatter_3d_mae.jpg"):
    ensure_dir_for_file(filename)
    if y_pred.ndim!=2:
        raise ValueError("y_pred must be 2D (N, out_dim)")
    _, out_dim= y_pred.shape
    fig, axes= plt.subplots(1,out_dim, figsize=(4*out_dim,4), squeeze=False)
    for i in range(out_dim):
        errors= np.abs(y_true[:,i]-y_pred[:,i])
        r2_val= r2_score(y_true[:,i], y_pred[:,i])
        ax= axes[0,i]
        sc= ax.scatter(y_true[:,i], y_pred[:,i], c=errors, alpha=0.5, cmap='ocean')
        min_val= min(y_true[:,i].min(), y_pred[:,i].min())
        max_val= max(y_true[:,i].max(), y_pred[:,i].max())
        ax.plot([min_val, max_val],[min_val, max_val],'r--', linewidth=1.5)

        if y_labels and i<len(y_labels):
            ax.set_title(f"{y_labels[i]} (MAE)\nR²={r2_val:.3f}")
            ax.set_xlabel(f"True {y_labels[i]}")
            ax.set_ylabel(f"Pred {y_labels[i]}")
        else:
            ax.set_title(f"Out {i} (MAE)\nR²={r2_val:.3f}")

        cbar= fig.colorbar(sc, ax=ax)
        cbar.set_label("Absolute Error")
    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()

def plot_residual_histogram(
        y_true, y_pred, y_labels=None,
        cmap_name="coolwarm",
        vmin=-45, vmax=45,
        filename="residual_hist_bottom.jpg"
):
    ensure_dir_for_file(filename)
    residuals = y_true - y_pred
    n_outputs = residuals.shape[1]

    fig, axes = plt.subplots(1, n_outputs, figsize=(4 * n_outputs, 4.5))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.88, wspace=0.3)

    num_bins = 30
    bins_array = np.linspace(vmin, vmax, num_bins + 1)

    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i in range(n_outputs):
        ax = axes[i] if n_outputs > 1 else axes
        hist_data, bin_edges, patches = ax.hist(
            residuals[:, i],
            bins=bins_array,
            alpha=0.9,
            edgecolor='none'
        )
        for b_idx, patch in enumerate(patches):
            bin_center = 0.5 * (bin_edges[b_idx] + bin_edges[b_idx + 1])
            patch.set_facecolor(cmap(norm(bin_center)))

        if y_labels and i < len(y_labels):
            ax.set_title(f"Residuals of {y_labels[i]}")
        else:
            ax.set_title(f"Output {i} Residual")

        ax.set_xlabel("Residual")
        ax.set_ylabel("Count")
        ax.set_xlim(vmin, vmax)

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

    plt.savefig(filename, dpi=700)
    plt.close()
    print(f"[plot_residual_histogram] => {filename}")


def plot_residual_kde(
    y_true, y_pred, y_labels=None,
    cmap_name="coolwarm",
    vmin=-45, vmax=45,
    filename="residual_kde_bottom.jpg"
):
    ensure_dir_for_file(filename)
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

    plt.savefig(filename, dpi=700)
    plt.close()
    print(f"[plot_residual_histogram] => {filename}")


def plot_rf_feature_importance_bar(feature_importances, feature_names, filename, top_k=20, threshold=0.05):
    """
    绘制 RF/Tree-based 模型的特征重要性条形图
    """
    ensure_dir_for_file(filename)

    if feature_importances is None or len(feature_importances) == 0:
        print("[WARN] No feature importances found!")
        return

    # 获取前 top_k 个最重要的特征
    sorted_idx = np.argsort(feature_importances)[::-1]  # 降序排序
    top_idx = sorted_idx[:top_k]
    top_feats = [feature_names[i] for i in top_idx]
    top_imps = feature_importances[top_idx]

    colors = ["red" if imp > threshold else "blue" for imp in top_imps]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(top_imps)), top_imps, align='center', color=colors)
    ax.set_yticks(range(len(top_imps)))
    ax.set_yticklabels(top_feats, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Feature Importance (Top-{top_k})", fontsize=14, fontweight='bold')
    ax.axvspan(0, threshold, facecolor='lightgray', alpha=0.5)
    ax.axvline(threshold, color='red', linestyle='dashed', linewidth=2)

    legend_e = [
        Patch(facecolor="red", label=f"> {threshold}"),
        Patch(facecolor="blue", label=f"<= {threshold}")
    ]
    ax.legend(handles=legend_e, loc="lower right", fontsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()
    print(f"[INFO] Feature importance plotted => {filename}")


def plot_three_metrics_horizontal(metrics_data, save_name="three_metrics.jpg"):
    ensure_dir_for_file(save_name)
    model_names= list(metrics_data.keys())
    mse_vals= [metrics_data[m]["MSE"] for m in model_names]
    mae_vals= [metrics_data[m]["MAE"] for m in model_names]
    r2_vals=  [metrics_data[m]["R2"]  for m in model_names]

    fig, axes= plt.subplots(1,3, figsize=(16,5))

    def plot_hbar_with_mean(ax, model_names, values, subplot_label, metric_label, bigger_is_better=False):
        arr= np.array(values)
        if bigger_is_better:
            best_idx= arr.argmax()
            worst_idx= arr.argmin()
        else:
            best_idx= arr.argmin()
            worst_idx= arr.argmax()

        colors= []
        for i,vv in enumerate(arr):
            if i==best_idx:
                colors.append("red")
            elif i==worst_idx:
                colors.append("blue")
            else:
                colors.append("green")

        mean_val= arr.mean()
        y_positions= np.arange(len(arr))
        ax.barh(y_positions, arr, color=colors, alpha=0.8, height=0.4)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(model_names)
        ax.invert_yaxis()
        ax.text(-0.08,1.05, subplot_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=14, fontweight="bold")
        ax.set_title(metric_label, fontsize=14, fontweight='bold')

        for i,vv in enumerate(arr):
            ax.text(vv, i, f"{vv:.2f}", ha=("left" if vv>=0 else "right"),
                    va="center", fontsize=10)

        ax.axvline(mean_val, color='gray', linestyle='--', linewidth=2)
        xmin, xmax= sorted([0, mean_val])
        ax.axvspan(xmin, xmax, facecolor='gray', alpha=0.2)

        max_val= arr.max()
        min_val= arr.min()
        if min_val<0:
            ax.set_xlim(min_val*1.1, max_val*1.79)
        else:
            ax.set_xlim(0, max_val*1.79)

        legend_e= [
            Patch(facecolor="red", label="Best"),
            Patch(facecolor="blue", label="Worst"),
            Patch(facecolor="green", label="Ordinary"),
            Patch(facecolor="gray", alpha=0.2, label="Under Mean"),
        ]
        ax.legend(handles=legend_e, loc="lower right")

    # (a) MSE
    plot_hbar_with_mean(axes[0], model_names, mse_vals, "(a)", "MSE (Lower=Better)", bigger_is_better=False)
    # (b) MAE
    plot_hbar_with_mean(axes[1], model_names, mae_vals, "(b)", "MAE (Lower=Better)", bigger_is_better=False)
    # (c) R2
    plot_hbar_with_mean(axes[2], model_names, r2_vals, "(c)", "R2 (Higher=Better)", bigger_is_better=True)

    plt.tight_layout()
    plt.savefig(save_name, dpi=700)
    plt.close()
    print(f"[plot_three_metrics_horizontal] => {save_name}")

def plot_overfitting_horizontal(overfit_data, save_name="overfitting_horizontal.jpg"):
    ensure_dir_for_file(save_name)
    model_names= list(overfit_data.keys())
    msr_vals= [overfit_data[m]["MSE_ratio"] for m in model_names]
    r2d_vals= [overfit_data[m]["R2_diff"]   for m in model_names]

    fig, axes= plt.subplots(1,2, figsize=(12,5))

    def plot_hbar_threshold(ax, model_names, values, subplot_label, metric_label,
                            bigger_is_better=False, threshold_h=0.5, threshold_l=0.0):
        arr= np.array(values)
        if bigger_is_better:
            best_idx= arr.argmax()
            worst_idx= arr.argmin()
        else:
            best_idx= arr.argmin()
            worst_idx= arr.argmax()

        colors= []
        for i,vv in enumerate(arr):
            if i==best_idx:
                colors.append("red")
            elif i==worst_idx:
                colors.append("blue")
            else:
                colors.append("green")

        y_positions= np.arange(len(arr))
        ax.barh(y_positions, arr, color=colors, alpha=0.8, height=0.4)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(model_names)
        ax.invert_yaxis()
        ax.text(-0.08,1.05, subplot_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=14, fontweight="bold")
        ax.set_title(metric_label, fontsize=14, fontweight='bold')

        for i,vv in enumerate(arr):
            ax.text(vv, i, f"{vv:.2f}", ha=("left" if vv>=0 else "right"),
                    va="center", fontsize=10)

        if threshold_l==0.0:
            ax.axvspan(threshold_l, threshold_h, facecolor='gray', alpha=0.2)
            legend_e= [ Patch(facecolor='gray', alpha=0.2, label="Acceptable") ]
        else:
            ax.axvspan(0, threshold_l, facecolor='gray', alpha=0.2)
            ax.axvspan(threshold_l, threshold_h, facecolor='lightcoral', alpha=0.3)
            ax.axvline(threshold_l, color='gray', linestyle='--', linewidth=2)
            ax.axvline(threshold_h, color='gray', linestyle='--', linewidth=2)
            legend_e= [
                Patch(facecolor='gray', alpha=0.2, label="Acceptable"),
                Patch(facecolor='lightcoral', alpha=0.3, label="Overfitting Risk")
            ]

        max_val= arr.max()
        min_val= arr.min()
        if min_val<0:
            ax.set_xlim(min_val*1.1, max_val*1.79)
        else:
            ax.set_xlim(0, max_val*1.79)

        legend_e.extend([
            Patch(facecolor="red", label="Best"),
            Patch(facecolor="blue", label="Worst"),
            Patch(facecolor="green", label="Ordinary")
        ])
        ax.legend(handles=legend_e, loc="lower right")

    plot_hbar_threshold(axes[0], model_names, msr_vals, "(a)", "MSE Ratio (Val/Train)\n(Lower=Better)",
                        bigger_is_better=False, threshold_h=10)
    plot_hbar_threshold(axes[1], model_names, r2d_vals, "(b)", "R2 diff (Train - Val)\n(Lower=Better)",
                        bigger_is_better=False, threshold_h=0.2, threshold_l=0.15)

    plt.tight_layout()
    plt.savefig(save_name, dpi=700)
    plt.close()
    print(f"[plot_overfitting_horizontal] => {save_name}")

# --------------- 原始数据分析 ---------------
def plot_kde_distribution(df, columns, filename):
    ensure_dir_for_file(filename)
    fig, axes= plt.subplots(1,len(columns), figsize=(5*len(columns),5))
    if len(columns)==1:
        axes= [axes]
    for i,col in enumerate(columns):
        ax= axes[i]
        if col not in df.columns:
            ax.text(0.5,0.5,f"'{col}' not in df", ha='center', va='center')
            continue

        sns.kdeplot(df[col], ax=ax, fill=False, color="black", clip=(df[col].min(), df[col].max()))
        lines= ax.get_lines()
        if not lines:
            ax.set_title(f"No Data for {col}")
            continue

        line= lines[-1]
        x_plot, y_plot= line.get_xdata(), line.get_ydata()
        idxsort= np.argsort(x_plot)
        x_plot, y_plot= x_plot[idxsort], y_plot[idxsort]

        vmin = max(np.min(x_plot), df[col].min())  # 发现负数
        vmax = min(np.max(x_plot), df[col].max())
        cmap= cm.get_cmap("coolwarm")
        norm= mcolors.Normalize(vmin=vmin, vmax=vmax)

        for j in range(len(x_plot)-1):
            x0, x1= x_plot[j], x_plot[j+1]
            y0, y1= y_plot[j], y_plot[j+1]
            color= cmap(norm((x0+x1)*0.5))
            verts= np.array([
                [x0,0],
                [x0,y0],
                [x1,y1],
                [x1,0]
            ])
            poly= PolyCollection([verts], facecolors=[color], edgecolor='none', alpha=0.6)
            ax.add_collection(poly)

        ax.set_title(f"KDE of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.set_xlim(df[col].min(), df[col].max())

        sm= cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb= plt.colorbar(sm, ax=ax)
        cb.set_label("Value Range", fontweight='bold', fontsize=12)
        cb.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()

def plot_catalyst_size_vs_product(df, filename):
    ensure_dir_for_file(filename)
    fig, axes= plt.subplots(2,2, figsize=(15,10), sharex=True)
    products= ['H2','CO','C1','C2+']
    for i,product in enumerate(products):
        ax= axes[i//2, i%2]
        needed= ['Particle size (nm)','Active metal',product]
        if all(c in df.columns for c in needed):
            sns.scatterplot(x='Particle size (nm)', y=product, hue='Active metal', data=df, ax=ax, alpha=0.7)
            ax.set_title(f'Particle size vs {product} Yield')
            ax.set_xlabel('Particle size (nm)')
            ax.set_ylabel(f'{product} Yield (%)')
        else:
            ax.text(0.5,0.5,f"Cols not found => {product}", ha='center', va='center')
    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()

def plot_potential_vs_product_by_electrolyte(df, filename):
    ensure_dir_for_file(filename)
    fig, axes= plt.subplots(2,2, figsize=(15,10), sharex=True)
    products= ['H2','CO','C1','C2+']
    for i,product in enumerate(products):
        ax= axes[i//2, i%2]
        needed= ['Potential (V vs. RHE)','Electrode support',product]
        if all(c in df.columns for c in needed):
            sns.scatterplot(x='Potential (V vs. RHE)', y=product, hue='Electrode support',
                            data=df, ax=ax, alpha=0.7)
            ax.set_title(f'Potential vs {product}')
            ax.set_xlabel('Potential (V vs. RHE)')
            ax.set_ylabel(f'{product} Yield (%)')
        else:
            ax.text(0.5,0.5,f"Cols not found => {product}", ha='center', va='center')
    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()

def plot_product_distribution_by_catalyst_and_potential(df, filename):
    ensure_dir_for_file(filename)
    fig, axes= plt.subplots(1,4, figsize=(20,5))
    products= ['H2','CO','C1','C2+']
    if 'Potential (V vs. RHE)' in df.columns:
        df['Potential_bin']= pd.cut(df['Potential (V vs. RHE)'], bins=5)
    else:
        df['Potential_bin']= "Unknown"
    for i,product in enumerate(products):
        ax= axes[i]
        needed= ['Active metal', product, 'Potential_bin']
        if all(c in df.columns for c in needed):
            sns.boxplot(x='Active metal', y=product, hue='Potential_bin', data=df, ax=ax)
            ax.set_title(f'{product} by ActiveMetal & Potential')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5,0.5,f"Cols not found => {product}", ha='center', va='center')
    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()

def plot_product_vs_potential_bin(df, filename):
    ensure_dir_for_file(filename)
    products= ['H2','CO','C1','C2+']
    if 'Potential (V vs. RHE)' not in df.columns:
        print("[WARN] no Potential => skip")
        return
    df['Potential bin custom']= pd.cut(df['Potential (V vs. RHE)'], bins=5)
    fig, axes= plt.subplots(1,4, figsize=(20,5))
    for i,product in enumerate(products):
        ax= axes[i]
        if product not in df.columns:
            ax.text(0.5,0.5,f"No col => {product}", ha='center', va='center')
            continue
        sns.boxplot(x='Potential bin custom', y=product, data=df, ax=ax)
        ax.set_title(f"{product} vs Potential Bin")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()

def plot_product_vs_shape(df, filename):
    ensure_dir_for_file(filename)
    products= ['H2','CO','C1','C2+']
    if 'Shape' not in df.columns:
        print("[WARN] no 'Shape' => skip")
        return
    fig, axes= plt.subplots(1,4, figsize=(20,5))
    for i,product in enumerate(products):
        ax= axes[i]
        if product not in df.columns:
            ax.text(0.5,0.5,f"No col => {product}", ha='center', va='center')
            continue
        sns.boxplot(x='Shape', y=product, data=df, ax=ax)
        ax.set_title(f"{product} vs Shape")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()

def plot_product_vs_catalyst(df, filename):
    ensure_dir_for_file(filename)
    products= ['H2','CO','C1','C2+']
    if 'Active metal' not in df.columns:
        print("[WARN] no 'Active metal' => skip")
        return
    fig, axes= plt.subplots(1,4, figsize=(20,5))
    for i,product in enumerate(products):
        ax= axes[i]
        if product not in df.columns:
            ax.text(0.5,0.5,f"No col => {product}", ha='center', va='center')
            continue
        sns.boxplot(x='Active metal', y=product, data=df, ax=ax)
        ax.set_title(f"{product} vs Active metal")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()

def plot_potential_vs_product(df, filename):
    ensure_dir_for_file(filename)
    products= ['H2','CO','C1','C2+']
    if 'Potential (V vs. RHE)' not in df.columns:
        print("[WARN] no 'Potential (V vs. RHE)' => skip")
        return
    plt.figure(figsize=(7,6))
    for product in products:
        if product in df.columns:
            plt.scatter(df['Potential (V vs. RHE)'], df[product], label=product, alpha=0.7)
    plt.title("Potential vs Products")
    plt.xlabel("Potential (V vs. RHE)")
    plt.ylabel("Yield (%)")
    plt.legend()
    plt.savefig(filename, dpi=700)
    plt.close()

# ================ 2D Heatmap 绘制 =====================
def plot_2d_heatmap_from_npy(grid_x, grid_y, heatmap_pred,
                             out_dir,
                             x_label="X-axis",
                             y_label="Y-axis",
                             y_col_names=None,
                             stats_dict=None,
                             colorbar_extend_ratio=0.25):
    """
    heatmap_pred shape=(H,W,out_dim).
    若 stats_dict 和 y_col_names 对应上，则使用 stats_dict[y_name]["min"/"max"] 做颜色范围；
    否则自动从 heatmap_pred 的数据范围 (z_.min, z_.max) 中获取。
    """
    os.makedirs(out_dir, exist_ok=True)
    H, W, out_dim = heatmap_pred.shape

    for odx in range(out_dim):
        z_ = heatmap_pred[:, :, odx]

        # 自动数据范围
        auto_min, auto_max = z_.min(), z_.max()

        # 若 stats_dict 里能找到 y_col_names[odx]，则使用统计信息，否则用数据本身
        if (stats_dict is not None) and (y_col_names is not None) \
           and (odx < len(y_col_names)) and (y_col_names[odx] in stats_dict):
            real_min = stats_dict[y_col_names[odx]]["min"]
            real_max = stats_dict[y_col_names[odx]]["max"]

            # 旧逻辑: clamp 到 [0,100] + extend
            vmin_ = max(0, real_min * (1 - colorbar_extend_ratio))
            vmax_ = min(100, real_max * (1 + colorbar_extend_ratio))
        else:
            # 无 stats => 自动用数据范围
            vmin_ = auto_min
            vmax_ = auto_max

        # 构造颜色映射
        norm_ = mcolors.Normalize(vmin=vmin_, vmax=vmax_)

        plt.figure(figsize=(6,5))
        cm_ = plt.pcolormesh(grid_x, grid_y, z_, shading='auto', cmap='viridis', norm=norm_)
        cb_ = plt.colorbar(cm_)

        if y_col_names and odx < len(y_col_names):
            cb_.set_label(y_col_names[odx], fontsize=12)
            plt.title(f"2D Heatmap of {y_col_names[odx]}", fontsize=14)
        else:
            cb_.set_label(f"Output_{odx}", fontsize=12)
            plt.title(f"2D Heatmap - out {odx}", fontsize=14)

        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)

        out_jpg = os.path.join(out_dir, f"heatmap_output_{odx+1}.jpg")
        plt.savefig(out_jpg, dpi=700, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 2D Heatmap saved => {out_jpg}")


def plot_3d_surface_from_heatmap(grid_x, grid_y, heatmap_pred,
                                 out_dir,
                                 x_label="X-axis",
                                 y_label="Y-axis",
                                 y_col_names=None,
                                 stats_dict=None,
                                 colorbar_extend_ratio=0.25,
                                 cmap_name="viridis"):
    """
    绘制三维曲面图 (surface plot)。若 stats_dict 存在，则用统计区间；否则用 heatmap_pred 的数据区间。
    """

    os.makedirs(out_dir, exist_ok=True)
    H, W, out_dim = heatmap_pred.shape

    for odx in range(out_dim):
        Z = heatmap_pred[:, :, odx]

        # 自动数据范围
        auto_min, auto_max = Z.min(), Z.max()

        if (stats_dict is not None) and (y_col_names is not None) \
           and (odx < len(y_col_names)) and (y_col_names[odx] in stats_dict):
            real_min = stats_dict[y_col_names[odx]]["min"]
            real_max = stats_dict[y_col_names[odx]]["max"]
            vmin_ = max(0, real_min * (1 - colorbar_extend_ratio))
            vmax_ = min(100, real_max * (1 + colorbar_extend_ratio))
        else:
            vmin_ = auto_min
            vmax_ = auto_max

        norm_ = mcolors.Normalize(vmin=vmin_, vmax=vmax_)
        cmap_ = plt.get_cmap(cmap_name)

        # 将 Z 映射到 RGBA
        Z_flat = Z.flatten()
        colors_rgba = cmap_(norm_(Z_flat))
        colors_rgba = colors_rgba.reshape((H, W, 4))

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(
            grid_x, grid_y, Z,
            facecolors=colors_rgba,
            rstride=1, cstride=1,
            linewidth=0, antialiased=False,
            shade=False
        )

        # colorbar
        sm = matplotlib.cm.ScalarMappable(norm=norm_, cmap=cmap_)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.1, aspect=15)

        if (y_col_names is not None) and (odx < len(y_col_names)):
            cb.set_label(y_col_names[odx], fontsize=12)
            ax.set_title(f"3D Surface of {y_col_names[odx]}", fontsize=14)
        else:
            cb.set_label(f"Output_{odx}", fontsize=12)
            ax.set_title(f"3D Surface - out {odx}", fontsize=14)

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_zlabel("Value", fontsize=12)
        ax.grid(False)

        out_jpg = os.path.join(out_dir, f"heatmap_3d_surface_output_{odx+1}.jpg")
        plt.savefig(out_jpg, dpi=700, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 3D Surface saved => {out_jpg}")


def plot_confusion_from_npy(confusion_pred,
                            row_labels, col_labels,
                            out_dir,
                            y_col_names=None,
                            stats_dict=None,
                            cell_scale=1/5,
                            colorbar_extend_ratio=0.25,
                            row_axis_name="Row Axis",
                            col_axis_name="Col Axis"):
    """
    confusion_pred shape=(n_rows,n_cols,out_dim)，MIMO 的“混淆矩阵”可视化。
    - 如果 stats_dict 中有该维度对应的统计信息，则用它来确定最小/最大值；
      否则根据该维度的最小值和最大值进行归一化。
    - 最多显示 4 个维度 (odx in [0..3])，并各画一个色条。
    """

    os.makedirs(out_dir, exist_ok=True)
    n_rows, n_cols, out_dim = confusion_pred.shape

    # 标签处理
    row_labels = [short_label(lbl) for lbl in row_labels]
    col_labels = [short_label(lbl) for lbl in col_labels]
    if y_col_names:
        y_col_names = [short_label(name) for name in y_col_names]

    # 最多显示前 4 个输出维度
    dim_used = min(4, out_dim)

    # 为每个维度计算归一化 (0..1)
    cmaps = [plt.get_cmap("Reds"), plt.get_cmap("Blues"),
             plt.get_cmap("Greens"), plt.get_cmap("Oranges")]
    norms = []

    for odx in range(dim_used):
        all_vals_dim = confusion_pred[:, :, odx]
        auto_min, auto_max = all_vals_dim.min(), all_vals_dim.max()

        # 如果 stats_dict 存在且含有该维度名称，则用它；否则直接用 auto_min, auto_max
        if (stats_dict is not None) and (y_col_names is not None) \
           and (odx < len(y_col_names)) and (y_col_names[odx] in stats_dict):
            real_min = stats_dict[y_col_names[odx]]["min"]
            real_max = stats_dict[y_col_names[odx]]["max"]
        else:
            real_min = auto_min
            real_max = auto_max

        # 将该维度数据归一化到 [0,1]
        confusion_pred[:, :, odx] = normalize_data(all_vals_dim, real_min, real_max)

        # 设置颜色归一化
        norm_ = mcolors.Normalize(vmin=0, vmax=1)
        norms.append(norm_)

    # 开始绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Confusion-like MIMO (No numeric)", fontsize=14)
    ax.set_aspect("equal", "box")
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.1)

    # 画网格
    for rr in range(n_rows + 1):
        ax.axhline(rr * cell_scale, color='black', linewidth=1)
    for cc in range(n_cols + 1):
        ax.axvline(cc * cell_scale, color='black', linewidth=1)

    # 画多三角区
    for i in range(n_rows):
        for j in range(n_cols):
            vals = confusion_pred[i, j, :]
            # 确定单元格四顶点
            BL = (j * cell_scale, i * cell_scale)
            BR = ((j + 1) * cell_scale, i * cell_scale)
            TL = (j * cell_scale, (i + 1) * cell_scale)
            TR = ((j + 1) * cell_scale, (i + 1) * cell_scale)
            Cx = j * cell_scale + cell_scale / 2
            Cy = i * cell_scale + cell_scale / 2

            for odx in range(dim_used):
                val_ = vals[odx]
                norm_ = norms[odx]
                color_ = cmaps[odx](norm_(val_))

                # 按 odx 决定画哪个三角
                if odx == 0:
                    poly = [TL, (Cx, Cy), TR]
                elif odx == 1:
                    poly = [TR, (Cx, Cy), BR]
                elif odx == 2:
                    poly = [BR, (Cx, Cy), BL]
                else:
                    poly = [BL, (Cx, Cy), TL]

                ax.add_patch(plt.Polygon(poly, facecolor=color_, alpha=0.9))

    # 设置坐标轴
    ax.set_xlim(0, n_cols * cell_scale)
    ax.set_ylim(0, n_rows * cell_scale)
    ax.invert_yaxis()

    ax.set_xticks([(j + 0.5) * cell_scale for j in range(n_cols)])
    ax.set_yticks([(i + 0.5) * cell_scale for i in range(n_rows)])
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=7.5)

    ax.set_ylabel(row_axis_name, fontsize=12)
    ax.set_xlabel(col_axis_name, fontsize=12)

    # 画 ColorBar（顶端）
    cbar_width = 0.21
    cbar_height = 0.02
    cbar_bottom = 0.93
    cbar_left_start = 0.08

    for odx in range(dim_used):
        sm = plt.cm.ScalarMappable(norm=norms[odx], cmap=cmaps[odx])
        sm.set_array([])
        cax_left = cbar_left_start + odx * cbar_width
        cax = fig.add_axes([cax_left, cbar_bottom, cbar_width, cbar_height])
        cb_ = plt.colorbar(sm, cax=cax, orientation='horizontal', pad=0.2)

        if (y_col_names is not None) and (odx < len(y_col_names)):
            short_lbl = y_col_names[odx]
        else:
            short_lbl = f"Out {odx}"

        cb_.set_label(short_lbl, fontsize=9, labelpad=2)
        # cb_.ax.tick_params(labelsize=8)
        # 移除 colorbar 的刻度标签
        cb_.set_ticks([])
        cb_.ax.xaxis.set_label_position('bottom')
        cb_.ax.xaxis.set_ticks_position('top')

    outfn = os.path.join(out_dir, "confusion_matrix_mimo.jpg")
    plt.savefig(outfn, dpi=700, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion => {outfn}")


def plot_3d_bars_from_confusion(confusion_pred,
                                row_labels, col_labels,
                                out_dir,
                                y_col_names=None,
                                stats_dict=None,
                                colorbar_extend_ratio=0.25,
                                cmap_name="viridis"):
    """
    绘制三维柱状图(Bar3D)的 “confusion-like” 图。
    - 若 stats_dict 存在且含有对应维度的统计范围，则用其 min/max；
      否则用该维度数据的最小值、最大值。
    - 将 x/y 刻度对准柱体中心，并使得刻度标签居中。
    - 每个维度单独输出一个 3D 柱状图。
    """

    os.makedirs(out_dir, exist_ok=True)
    n_rows, n_cols, out_dim = confusion_pred.shape

    # 标签处理
    row_labels = [short_label(lbl) for lbl in row_labels]
    col_labels = [short_label(lbl) for lbl in col_labels]
    if y_col_names:
        y_col_names = [short_label(name) for name in y_col_names]

    # 我们在此不限制维度个数，每个维度绘制一个图
    for odx in range(out_dim):
        all_vals_dim = confusion_pred[:, :, odx]
        auto_min, auto_max = all_vals_dim.min(), all_vals_dim.max()

        if (stats_dict is not None) and (y_col_names is not None) \
           and (odx < len(y_col_names)) and (y_col_names[odx] in stats_dict):
            real_min = stats_dict[y_col_names[odx]]["min"]
            real_max = stats_dict[y_col_names[odx]]["max"]
        else:
            real_min = auto_min
            real_max = auto_max

        # 归一化该维度数据到 [0,1]
        Z = normalize_data(all_vals_dim, real_min, real_max)

        norm_ = mcolors.Normalize(vmin=0, vmax=1)
        cmap_ = plt.get_cmap(cmap_name)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        dx = dy = 0.5

        x_vals, y_vals, z_vals = [], [], []
        dz_vals, facecolors = [], []

        for i in range(n_rows):
            for j in range(n_cols):
                val_ = Z[i, j]
                x_vals.append(j)
                y_vals.append(i)
                z_vals.append(0)
                dz_vals.append(val_)
                # 根据归一化后的值生成颜色
                facecolors.append(cmap_(norm_(val_)))

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        z_vals = np.array(z_vals)
        dz_vals = np.array(dz_vals)

        ax.bar3d(
            x_vals, y_vals, z_vals,
            dx, dy, dz_vals,
            color=facecolors, alpha=0.75, shade=True
        )

        ax.grid(False)

        # 让刻度居中对齐柱体
        ax.set_xticks(np.arange(n_cols) + dx / 2)
        ax.set_yticks(np.arange(n_rows) + dy / 2)

        # X 轴标签：旋转 45 度并右对齐
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=10)
        # Y 轴标签：根据需求选择合适的旋转
        ax.set_yticklabels(row_labels, rotation=-15, ha='left', va='center', fontsize=10)

        # 仅保留 Z 轴名称
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("Value", fontsize=12)

        # 颜色条
        sm = plt.cm.ScalarMappable(norm=norm_, cmap=cmap_)
        sm.set_array([])  # 不对应具体数组，只作颜色映射
        cb_ = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1, aspect=15)

        # 标题
        if y_col_names and odx < len(y_col_names):
            var_name = y_col_names[odx]
            ax.set_title(f"3D Bars Confusion - {var_name}", fontsize=14)
            cb_.set_label(var_name, fontsize=12)
        else:
            var_name = f"Output_{odx}"
            ax.set_title(f"3D Bars Confusion - out {odx}", fontsize=14)
            cb_.set_label(var_name, fontsize=12)

        out_jpg = os.path.join(out_dir, f"3d_bars_confusion_output_{odx+1}.jpg")
        plt.savefig(out_jpg, dpi=700, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 3D Bars Confusion saved => {out_jpg}")