#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference.py

- 读取 ./models/<model_type>/trained_model.pkl / best_ann.pt
- 读取 metadata.pkl (continuous_cols, onehot_groups, oh_index_map, observed_onehot_combos …)
- 只在实际出现过的 one‑hot 组合上做平均 (避免从未见过的组合)
- 输出 heatmap_pred.npy, confusion_pred.npy 等
  (可加权: sum_real += real_pred * freq; avg_real = sum_real / sum_freq)
"""

import yaml
import os
import numpy as np
import torch
import joblib
from tqdm import trange
from itertools import product   # 你原来的 import 保留
import json                     # 你原来的 import 保留

from data_preprocessing.scaler_utils import load_scaler, inverse_transform_output
from utils import get_model_dir, get_root_model_dir

# 各种模型
from models.model_ann import ANNRegression
from models.model_rf import RFRegression
from models.model_dt import DTRegression
from models.model_catboost import CatBoostRegression
from models.model_xgb import XGBRegression

from itertools import combinations


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)



# 放到 inference_main() 顶部、for-mtype 循环里 —— 在拿到 x_col_names 之后
def find_group_idx(keyword, groups, colnames):
    kw = keyword.lower()
    for idx, grp in enumerate(groups):
        if any(kw in colnames[c].lower() for c in grp):
            return idx
    return None

# --------------------------------------------------
#              按类型加载已训练模型
# --------------------------------------------------
def load_inference_model(model_type, config):
    csv_name = os.path.splitext(os.path.basename(config["data"]["path"]))[0]
    model_dir = get_model_dir(csv_name, model_type)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"[ERROR] Directory not found => {model_dir}")

    x_col_path = os.path.join(model_dir, "x_col_names.npy")
    y_col_path = os.path.join(model_dir, "y_col_names.npy")
    if not (os.path.exists(x_col_path) and os.path.exists(y_col_path)):
        raise FileNotFoundError("[ERROR] x_col_names.npy or y_col_names.npy not found.")

    x_col_names = list(np.load(x_col_path, allow_pickle=True))
    y_col_names = list(np.load(y_col_path, allow_pickle=True))

    # ---------- ANN ----------
    if model_type == "ANN":
        ann_cfg = config["model"]["ann_params"].copy()

        # 若 hidden_dims 不在 ann_cfg，就从 Optuna 最优参数里补
        if "hidden_dims" not in ann_cfg:
            optuna_dir = os.path.join("postprocessing", csv_name, "optuna", "ANN")
            best_params_path = os.path.join(optuna_dir, "best_params.pkl")
            if not os.path.exists(best_params_path):
                raise FileNotFoundError(f"[ERROR] {best_params_path} not found. Please run hyper‑parameter search first.")
            best_params = joblib.load(best_params_path)
            if isinstance(best_params.get("hidden_dims"), str):
                best_params["hidden_dims"] = tuple(int(x) for x in best_params["hidden_dims"].split(","))
            ann_cfg.update(best_params)
            print(f"[INFO] Updated ann_params from optuna: {ann_cfg}")

        net = ANNRegression(
            input_dim=len(x_col_names),
            output_dim=len(y_col_names),
            hidden_dims=ann_cfg["hidden_dims"],
            dropout=ann_cfg.get("dropout", 0.0),
            activation=ann_cfg.get("activation", "ReLU"),
            random_seed=ann_cfg.get("random_seed", 42)
        )

        ckpt_path = os.path.join(model_dir, "best_ann.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[ERROR] {ckpt_path} not found.")
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:  # 兼容旧版 torch
            state_dict = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(state_dict)
        net.eval()
        return net, x_col_names, y_col_names

    # ---------- 其余模型 ----------
    else:
        pkl_path = os.path.join(model_dir, "trained_model.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"[ERROR] {pkl_path} not found.")
        model = joblib.load(pkl_path)
        return model, x_col_names, y_col_names


def model_predict(model, X_2d):
    """统一预测接口：Torch / Sklearn / Booster"""
    if hasattr(model, "eval") and hasattr(model, "forward"):
        with torch.no_grad():
            out = model(torch.tensor(X_2d, dtype=torch.float32)).cpu().numpy()
    else:
        out = model.predict(X_2d)

    # --- 新增：保证 2D ---
    if out.ndim == 1:
        out = out.reshape(-1, 1)

    return out



def get_onehot_global_col_index(local_oh_index, oh_index_map):
    return oh_index_map[local_oh_index]

# ==============================================================
#            ① 复用原 2D 逻辑 → 包成函数
# ==============================================================

def heatmap_2d_inference(model, x_name, y_name,
                         stats_dict, x_col_names, numeric_cols_idx,
                         scaler_x, scaler_y,
                         observed_combos, oh_index_map,
                         outdir_m, n_points=50):
    """把你原先那段 2D 推断代码完整挪进来──除了 x_name/y_name 改成形参"""
    if (x_name not in stats_dict["continuous_cols"]
            or y_name not in stats_dict["continuous_cols"]):
        print(f"[WARN] {x_name}/{y_name} 不在连续列中 => 跳过 2D")
        return

    xinfo = stats_dict["continuous_cols"][x_name]
    yinfo = stats_dict["continuous_cols"][y_name]

    xv = np.linspace(xinfo["min"], xinfo["max"], n_points)
    yv = np.linspace(yinfo["min"], yinfo["max"], n_points)
    grid_x, grid_y = np.meshgrid(xv, yv)

    # —— baseline（连续列均值） ——
    base_vec = np.zeros(len(x_col_names))
    for cname, cstat in stats_dict["continuous_cols"].items():
        if cname in x_col_names:
            base_vec[x_col_names.index(cname)] = cstat["mean"]

    tmp = base_vec.reshape(1, -1)
    if scaler_x is not None:
        tmp[:, numeric_cols_idx] = scaler_x.transform(tmp[:, numeric_cols_idx])
    out_dim = model_predict(model, tmp).shape[-1]

    H, W = grid_x.shape
    heatmap_pred = np.zeros((H, W, out_dim))

    for i in trange(H, desc=f"2D({x_name},{y_name})", ncols=100):
        for j in range(W):
            vec = base_vec.copy()
            vec[x_col_names.index(x_name)] = grid_x[i, j]
            vec[x_col_names.index(y_name)] = grid_y[i, j]

            sum_real = np.zeros(out_dim)
            for oh_tuple, _ in observed_combos:
                tmpv = vec.copy()
                for local_idx, v01 in enumerate(oh_tuple):
                    gcol = get_onehot_global_col_index(local_idx, oh_index_map)
                    tmpv[gcol] = v01
                xin = tmpv.reshape(1, -1)
                if scaler_x is not None:
                    xin[:, numeric_cols_idx] = scaler_x.transform(xin[:, numeric_cols_idx])
                scaled = model_predict(model, xin)
                real   = inverse_transform_output(scaled, scaler_y)
                sum_real += real.squeeze()

            heatmap_pred[i, j, :] = np.maximum(sum_real / len(observed_combos), 0)

    # —— 保存 ——
    tag = f"{x_name}__{y_name}".replace(" ", "_").replace("/", "_")
    np.save(os.path.join(outdir_m, f"grid_x_{tag}.npy"), grid_x)
    np.save(os.path.join(outdir_m, f"grid_y_{tag}.npy"), grid_y)
    np.save(os.path.join(outdir_m, f"heatmap_pred_{tag}.npy"), heatmap_pred)
    print(f"[INFO] 2D heatmap ({x_name},{y_name}) saved → {outdir_m}")


# ==============================================================
#            ② 三变量 3D 推断（透明等值面用）
# ==============================================================

def heatmap_3d_inference(model, axes_names, stats_dict,
                         x_col_names, numeric_cols_idx,
                         scaler_x, scaler_y,
                         observed_combos, oh_index_map,
                         outdir_m, n_points=40):
    """axes_names = [x_name, y_name, z_name]"""
    x_name, y_name, z_name = axes_names

    def _mm(col):
        info = stats_dict["continuous_cols"][col]
        return info["min"], info["max"]

    xv = np.linspace(*_mm(x_name), n_points)
    yv = np.linspace(*_mm(y_name), n_points)
    zv = np.linspace(*_mm(z_name), n_points)
    grid_x, grid_y, grid_z = np.meshgrid(xv, yv, zv, indexing="ij")

    base_vec = np.zeros(len(x_col_names))
    for cname, cstat in stats_dict["continuous_cols"].items():
        if cname in x_col_names:
            base_vec[x_col_names.index(cname)] = cstat["mean"]

    tmp = base_vec.reshape(1, -1)
    if scaler_x is not None:
        tmp[:, numeric_cols_idx] = scaler_x.transform(tmp[:, numeric_cols_idx])
    out_dim = model_predict(model, tmp).shape[-1]

    H, W, D = grid_x.shape
    heatmap_pred = np.zeros((H, W, D, out_dim))

    for i in trange(H, desc="3DHeatmap-X", ncols=100):
        for j in range(W):
            for k in range(D):
                vec = base_vec.copy()
                vec[x_col_names.index(x_name)] = grid_x[i, j, k]
                vec[x_col_names.index(y_name)] = grid_y[i, j, k]
                vec[x_col_names.index(z_name)] = grid_z[i, j, k]

                sum_real = np.zeros(out_dim)
                for oh_tuple, _ in observed_combos:
                    tmpv = vec.copy()
                    for local_idx, v01 in enumerate(oh_tuple):
                        gcol = get_onehot_global_col_index(local_idx, oh_index_map)
                        tmpv[gcol] = v01
                    xin = tmpv.reshape(1, -1)
                    if scaler_x is not None:
                        xin[:, numeric_cols_idx] = scaler_x.transform(xin[:, numeric_cols_idx])
                    scaled = model_predict(model, xin)
                    real   = inverse_transform_output(scaled, scaler_y)
                    sum_real += real.squeeze()

                heatmap_pred[i, j, k, :] = np.maximum(sum_real / len(observed_combos), 0)

    np.save(os.path.join(outdir_m, "grid_x_3d.npy"), grid_x)
    np.save(os.path.join(outdir_m, "grid_y_3d.npy"), grid_y)
    np.save(os.path.join(outdir_m, "grid_z_3d.npy"), grid_z)
    np.save(os.path.join(outdir_m, "heatmap_pred_3d.npy"), heatmap_pred)
    print(f"[INFO] 3D heatmap saved → {outdir_m}")

# --------------------------------------------------
#                    主入口
# --------------------------------------------------
def inference_main():
    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    inf_models = config["inference"].get("models", [])
    if not inf_models:
        print("[INFO] No inference models => exit.")
        return

    # >>> PATCH: 先解析数据集名称 & metadata 路径 -----------------------------
    csv_path = config["data"]["path"]
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    root_model_dir = get_root_model_dir(csv_name)
    meta_path = os.path.join(root_model_dir, "metadata.pkl")
    # <<< PATCH ----------------------------------------------------------------

    if not os.path.exists(meta_path):
        print(f"[ERROR] metadata => {meta_path} missing.")
        return

    stats_dict = joblib.load(meta_path)

    observed_combos  = stats_dict["observed_onehot_combos"]
    oh_index_map     = stats_dict["oh_index_map"]
    numeric_cols_idx = stats_dict["numeric_cols_idx"]

    base_inf = os.path.join("postprocessing", csv_name, "inference")
    ensure_dir(base_inf)


    # =================================================
    #               循环每个模型做推断
    # =================================================
    for mtype in inf_models:
        print(f"\n=== Inference => {mtype} ===")
        outdir_m = os.path.join(base_inf, mtype)
        ensure_dir(outdir_m)

        try:
            model, x_col_names, y_col_names = load_inference_model(mtype, config)
        except FileNotFoundError as e:
            print(e)
            continue

        # --- scaler ---
        model_dir = get_model_dir(csv_name, mtype)
        sx_path = os.path.join(model_dir, f"scaler_x_{mtype}.pkl")
        sy_path = os.path.join(model_dir, f"scaler_y_{mtype}.pkl")
        scaler_x = load_scaler(sx_path) if os.path.exists(sx_path) else None
        scaler_y = load_scaler(sy_path) if os.path.exists(sy_path) else None

        # 校验 numeric 列一致性（仅当 scaler_x 存在）
        if scaler_x and len(numeric_cols_idx) != scaler_x.n_features_in_:
            raise RuntimeError(
                f"numeric_cols_idx ({len(numeric_cols_idx)}) 与 "
                f"scaler_x.n_features_in_ ({scaler_x.n_features_in_}) 不匹配！"
            )
        # ----------------------------------------------------------
        # 根据 heatmap_axes 的长度自动分支
        # ----------------------------------------------------------
        axes_names = config["inference"].get("heatmap_axes", [])
        dim_axes = len(axes_names)
        n_points = config["inference"].get("n_points", 50)

        # if dim_axes == 2:
        #     # 直接 2D
        #     heatmap_2d_inference(model,
        #                          axes_names[0], axes_names[1],
        #                          stats_dict, x_col_names, numeric_cols_idx,
        #                          scaler_x, scaler_y,
        #                          observed_combos, oh_index_map,
        #                          outdir_m, n_points)
        #
        # elif dim_axes == 3:
        #     # ① 先跑 C(3,2) 三张 2D
        #     for x_name, y_name in combinations(axes_names, 2):
        #         heatmap_2d_inference(model,
        #                              x_name, y_name,
        #                              stats_dict, x_col_names, numeric_cols_idx,
        #                              scaler_x, scaler_y,
        #                              observed_combos, oh_index_map,
        #                              outdir_m, n_points)
        #     # ② 再跑 3D
        #     heatmap_3d_inference(model,
        #                          axes_names,
        #                          stats_dict, x_col_names, numeric_cols_idx,
        #                          scaler_x, scaler_y,
        #                          observed_combos, oh_index_map,
        #                          outdir_m, n_points)
        # else:
        #     print(f"[WARN] heatmap_axes={axes_names} (维数={dim_axes}) 非 2/3，已跳过连续变量可视化。")

        # =================================================
        #               B) Confusion‑like 输出
        # =================================================
        if len(stats_dict["onehot_groups"]) < 2:
            print("[WARN] Not enough onehot groups => skip confusion.")
            continue

        row_kw = config["inference"]["confusion_axes"]["row_name"]
        col_kw = config["inference"]["confusion_axes"]["col_name"]

        row_idx = find_group_idx(row_kw, stats_dict["onehot_groups"], x_col_names)
        col_idx = find_group_idx(col_kw, stats_dict["onehot_groups"], x_col_names)

        if (row_idx is None) or (col_idx is None):
            print("[WARN] 指定的 row/col 关键字没找到 —— 退回前两组")
            grpA, grpB = stats_dict["onehot_groups"][:2]
        else:
            grpA = stats_dict["onehot_groups"][row_idx]
            grpB = stats_dict["onehot_groups"][col_idx]

        base_vec = np.zeros(len(x_col_names), dtype=float)
        for cname, cstat in stats_dict["continuous_cols"].items():
            if cname in x_col_names:
                base_vec[x_col_names.index(cname)] = cstat["mean"]

        tmp = base_vec.reshape(1, -1)
        if scaler_x is not None:
            tmp[:, numeric_cols_idx] = scaler_x.transform(tmp[:, numeric_cols_idx])
        outdim = model_predict(model, tmp).shape[-1]

        confusion_pred = np.zeros((len(grpA), len(grpB), outdim), dtype=float)

        for i in trange(len(grpA), desc="Confusion Rows", ncols=100):
            for j in range(len(grpB)):
                sum_real = np.zeros(outdim)
                for oh_tuple, _ in observed_combos:
                    tmpv = base_vec.copy()

                    for local_idx, v01 in enumerate(oh_tuple):
                        tmpv[get_onehot_global_col_index(local_idx, oh_index_map)] = v01
                    # 指定当前 A/B 组合
                    tmpv[grpA] = 0
                    tmpv[grpB] = 0
                    tmpv[grpA[i]] = 1
                    tmpv[grpB[j]] = 1

                    tmp_inp = tmpv.reshape(1, -1)
                    if scaler_x is not None:
                        tmp_inp[:, numeric_cols_idx] = scaler_x.transform(tmp_inp[:, numeric_cols_idx])
                    scaled_pred = model_predict(model, tmp_inp)
                    real_pred = inverse_transform_output(scaled_pred, scaler_y)
                    sum_real += real_pred.squeeze()

                confusion_pred[i, j, :] = sum_real / len(observed_combos)
        # 循环结束之后（np.save 之前）插入
        v_min = confusion_pred.min()
        v_max = confusion_pred.max()
        eps = 1e-12  # 防除零
        confusion_norm = (confusion_pred - v_min) / (v_max - v_min + eps)

        # 新增一份归一化后的矩阵，名字示例：
        np.save(os.path.join(outdir_m, "confusion_pred_norm.npy"), confusion_norm)
        print(f"[INFO] confusion saved => {outdir_m}")


if __name__ == "__main__":
    inference_main()
