import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ========== 全局字体 & 样式 ==========
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['axes.labelsize'] = 13
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titlesize'] = 15
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 12

SAVE_DIR = "./evaluation/figures"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_hbar_with_mean(
    ax,
    model_names,
    values,
    subplot_label="(a)",
    metric_label="Metric",
    bigger_is_better=False,
    width=0.4
):
    """
    绘制横向条形图(barh):
      - 纵坐标：模型名称
      - 横坐标：对应数值
      - 最优 => 红色, 最差 => 蓝色, 普通 => 绿色
      - 均值 => 灰色虚线 + [0, mean] 灰色填充
      - 子图左上角加 subplot_label (如"(a)")
      - 标题里显示 metric_label
      - bar宽度可调(默认0.4) 使其更紧凑
      - 为防止图例覆盖，自动把 xlim 设到最大值的1.75倍左右
    """

    arr = np.array(values)

    # 找最优 / 最差
    if bigger_is_better:
        best_idx = arr.argmax()
        worst_idx = arr.argmin()
    else:
        best_idx = arr.argmin()
        worst_idx = arr.argmax()

    # 颜色
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
    bars = ax.barh(
        y_positions,
        arr,
        color=colors,
        alpha=0.8,
        height=width  # 横向bar是用 `height` 控制粗细，默认 if barh, width->height
    )

    # y刻度 = 模型名称
    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_names)
    ax.invert_yaxis()  # 让第0个在顶部

    # 子图左上角 (a)/ (b)/ (c)
    ax.text(
        -0.08, 1.05,
        subplot_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold"
    )

    # 标题 (例如: "MSE (Lower is better)")
    ax.set_title(metric_label, fontsize=14, fontweight='bold')

    # 在每个bar末端加数值标签
    for i, v in enumerate(arr):
        ax.text(
            v, i,
            f"{v:.2f}",
            ha=("left" if v >= 0 else "right"),
            va="center",
            fontsize=10
        )

    # 均值 => 灰色虚线
    ax.axvline(mean_val, color='gray', linestyle='--', linewidth=2)
    # 填充 [0, mean_val]
    xmin, xmax = sorted([0, mean_val])
    ax.axvspan(xmin, xmax, facecolor='gray', alpha=0.2)

    # 设定 xlim 给图例留空间 (加一个系数1.75)
    max_val = arr.max()
    min_val = arr.min()
    if min_val < 0:
        # 若有负值再灵活处理
        ax.set_xlim(min_val * 1.1, max_val * 1.7)
    else:
        ax.set_xlim(0, max_val * 1.7)

    # 图例
    legend_elements = [
        Patch(facecolor="red", label="Best"),
        Patch(facecolor="blue", label="Worst"),
        Patch(facecolor="green", label="Ordinary"),
        Patch(facecolor="gray", alpha=0.2, label="Under Mean"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

def plot_three_metrics_horizontal(
    metrics_data,
    save_name="three_metrics_horizontal.jpg"
):
    """
    画一行三张横向条形图:
      (a) MSE (Lower is better)
      (b) MAE (Lower is better)
      (c) R2  (Higher is better)
    不加总标题, 图整体更紧凑.
    """
    ensure_dir(SAVE_DIR)
    model_names = list(metrics_data.keys())

    # 取出 MSE, MAE, R2
    mse_vals = [metrics_data[m]["MSE"] for m in model_names]
    mae_vals = [metrics_data[m]["MAE"] for m in model_names]
    r2_vals  = [metrics_data[m]["R2"]  for m in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) MSE => 越小越好
    plot_hbar_with_mean(
        ax=axes[0],
        model_names=model_names,
        values=mse_vals,
        subplot_label="(a)",
        metric_label="MSE (Lower is better)",
        bigger_is_better=False,
        width=0.4
    )

    # (b) MAE => 越小越好
    plot_hbar_with_mean(
        ax=axes[1],
        model_names=model_names,
        values=mae_vals,
        subplot_label="(b)",
        metric_label="MAE (Lower is better)",
        bigger_is_better=False,
        width=0.4
    )

    # (c) R2 => 越大越好
    plot_hbar_with_mean(
        ax=axes[2],
        model_names=model_names,
        values=r2_vals,
        subplot_label="(c)",
        metric_label="R2 (Higher is better)",
        bigger_is_better=True,
        width=0.4
    )

    # 紧凑布局
    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot_three_metrics_horizontal] Saved => {out_path}")

# ============== 测试示例 ==============
if __name__ == "__main__":
    metrics_data = {
        "XGB":      {"MSE": 70.3673, "MAE": 6.1343, "R2": 0.7811},
        "RF":       {"MSE": 80.1752, "MAE": 6.6119, "R2": 0.7420},
        "DT":       {"MSE": 196.9244,"MAE": 10.3842,"R2": 0.3429},
        "CatBoost": {"MSE": 100.5415,"MAE": 7.5529, "R2": 0.6650}
    }

    plot_three_metrics_horizontal(metrics_data, "three_metrics_horizontal.jpg")
