#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py

- 读取 CSV (前14列)
- 得到 X, Y, numeric_cols_idx, x_col_names, y_col_names, observed_combos, onehot_groups, oh_index_map
- 训练模型并保存
- 将 observed_combos, onehot_groups, oh_index_map 等一起存进 metadata.pkl
"""

import yaml
import os
import numpy as np
import torch
import joblib
import copy
import shap
from catboost import Pool
import optuna
import shutil  # 用于复制调参结果到 evaluation 目录
from utils import get_model_dir, get_root_model_dir
import json                    # 需要写 ann_meta
from data_preprocessing.my_dataset import MyDataset

from data_preprocessing.data_loader import (
    load_dataset,
    load_raw_data_for_correlation,
    extract_data_statistics
)
from data_preprocessing.data_split import split_data
from data_preprocessing.scaler_utils import (
    standardize_data, inverse_transform_output, save_scaler
)

# 各种模型
from models.model_ann import ANNRegression
from models.model_rf import RFRegression
from models.model_dt import DTRegression
from models.model_catboost import CatBoostRegression
from models.model_xgb import XGBRegression

# 训练 & 评估
from losses.torch_losses import get_torch_loss_fn
from trainers.train_torch import train_torch_model_dataloader
from trainers.train_sklearn import train_sklearn_model
from evaluation.metrics import compute_regression_metrics, compute_mixed_metrics

from sklearn.model_selection import KFold
import pandas as pd            # ← 写在已有 import 区
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils import ensure_dir
import re                 # ← 处理列名用
from itertools import chain


# -----------------------------------------------------------------------
# helper ranges
# -----------------------------------------------------------------------
def _safe_float_range(cfg, lo_key, hi_key, min_bump=1.01):
    low, high = float(cfg[lo_key]), float(cfg[hi_key])
    if low > high:
        low, high = high, low
    if low == high:
        # 特别处理 0：乘 min_bump 仍是 0
        high = low + (1e-8 if low == 0 else low * (min_bump - 1))
    return low, high

def _safe_int_range(cfg, lo_key, hi_key):
    low, high = int(cfg[lo_key]), int(cfg[hi_key])
    if low > high:
        low, high = high, low
    if low == high:
        high = low + 1
    return low, high

def _to_2d(arr: np.ndarray) -> np.ndarray:
    """把 (n,) 向量统一 reshape 成 (n,1)。已是 2-D 的直接返回。"""
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr

# -----------------------------------------------------------------------
# main tuner
# -----------------------------------------------------------------------
def tune_model(model_type, config, X, Y,
               numeric_cols_idx, x_col_names, y_col_names,
               random_seed):

    # 1) split & standardize --------------------------------------------------
    X_train, X_val, Y_train, Y_val = split_data(
        X, Y, test_size=config["data"]["test_size"], random_state=random_seed
    )
    (X_train_s, X_val_s, sx), (Y_train_s, Y_val_s, sy) = standardize_data(
        X_train, X_val, Y_train, Y_val,
        do_input = config["preprocessing"]["standardize_input"],
        do_output= config["preprocessing"]["standardize_output"],
        numeric_cols_idx=numeric_cols_idx
    )

    # 2) Optuna objective -----------------------------------------------------
    def objective(trial):

        # -------- build & train each model -----------------------------------
        if model_type == "ANN":
            base = config["optuna"]["ann_params"]
            dims_choices = [",".join(map(str, d)) for d in base["hidden_dims_choices"]]
            hidden_dims = tuple(int(s) for s in trial.suggest_categorical(
                                "hidden_dims", dims_choices).split(","))

            d_lo,d_hi   = _safe_float_range(base, "dropout_min","dropout_max")
            lr_lo,lr_hi = _safe_float_range(base, "learning_rate_min","learning_rate_max")
            wd_lo,wd_hi = _safe_float_range(base, "weight_decay_min","weight_decay_max")

            dropout      = trial.suggest_float("dropout", d_lo, d_hi)
            lr           = trial.suggest_float("learning_rate",  lr_lo, lr_hi, log=True)
            weight_decay = trial.suggest_float("weight_decay",   wd_lo, wd_hi, log=True)
            batch_sz     = trial.suggest_categorical("batch_size", base["batch_size_choices"])
            optim        = trial.suggest_categorical("optimizer",  base["optimizer_choices"])
            actv         = trial.suggest_categorical("activation", base["activation_choices"])
            epochs       = base.get("tuning_epochs", 100)

            model_instance = ANNRegression(
                input_dim=X_train_s.shape[1],
                # output_dim=config["data"].get("output_len", 4),   #changed
                output_dim=config["data"]["output_len"],
                hidden_dims=hidden_dims,
                dropout=dropout,
                activation=actv,
                random_seed=random_seed
            )

            loss_fn = get_torch_loss_fn(config["loss"]["type"])

            train_ds, val_ds = MyDataset(X_train_s, Y_train_s), MyDataset(X_val_s, Y_val_s)

            model_instance, _, _ = train_torch_model_dataloader(
                model_instance, train_ds, val_ds,
                loss_fn=loss_fn,
                epochs=epochs,
                batch_size=batch_sz,
                lr=lr, weight_decay=weight_decay,
                checkpoint_path=None,
                log_interval=base.get("log_interval", 5),
                early_stopping=base.get("early_stopping", True),
                patience=base.get("patience", 10),
                optimizer_name=optim
            )

            # ---- 推断（关闭随机正则化） ----
            model_instance.eval()
            dev = next(model_instance.parameters()).device
            with torch.no_grad():
                pred_val   = model_instance(torch.tensor(X_val_s,   dtype=torch.float32, device=dev)).cpu().numpy()
                pred_train = model_instance(torch.tensor(X_train_s, dtype=torch.float32, device=dev)).cpu().numpy()

        elif model_type == "RF":
            base = config["optuna"]["rf_params"]
            n_lo,n_hi   = _safe_int_range  (base,"n_estimators_min","n_estimators_max")
            d_lo,d_hi   = _safe_int_range  (base,"max_depth_min",   "max_depth_max")
            c_lo,c_hi   = _safe_float_range(base,"ccp_alpha_min",   "ccp_alpha_max")
            l_lo,l_hi   = _safe_int_range  (base,"min_samples_leaf_min","min_samples_leaf_max")

            model_instance = RFRegression(
                n_estimators    = trial.suggest_int ("n_estimators", n_lo, n_hi),
                max_depth       = trial.suggest_int ("max_depth",    d_lo, d_hi),
                ccp_alpha       = trial.suggest_float("ccp_alpha",   c_lo, c_hi),
                min_samples_leaf= trial.suggest_int ("min_samples_leaf", l_lo, l_hi),
                random_state=random_seed
            )
            model_instance = train_sklearn_model(model_instance, X_train_s, Y_train_s)
            pred_train, pred_val = model_instance.predict(X_train_s), model_instance.predict(X_val_s)

        elif model_type == "DT":
            base = config["optuna"]["dt_params"]
            d_lo,d_hi = _safe_int_range  (base,"max_depth_min","max_depth_max")
            c_lo,c_hi = _safe_float_range(base,"ccp_alpha_min","ccp_alpha_max")

            model_instance = DTRegression(
                max_depth   = trial.suggest_int  ("max_depth", d_lo, d_hi),
                ccp_alpha   = trial.suggest_float("ccp_alpha", c_lo, c_hi),
                random_state= config["model"]["dt_params"]["random_state"]
            )
            model_instance = train_sklearn_model(model_instance, X_train_s, Y_train_s)
            pred_train, pred_val = model_instance.predict(X_train_s), model_instance.predict(X_val_s)

        elif model_type == "CatBoost":
            base = config["optuna"]["catboost_params"]
            it_lo,it_hi = _safe_int_range  (base,"iterations_min","iterations_max")
            lr_lo,lr_hi = _safe_float_range(base,"learning_rate_min","learning_rate_max")
            dep_lo,dep_hi = _safe_int_range(base,"depth_min","depth_max")
            l2_lo,l2_hi = _safe_float_range(base,"l2_leaf_reg_min","l2_leaf_reg_max")
            es_rounds = config["model"]["catboost_params"].get("early_stopping_rounds", 50)

            model_instance = CatBoostRegression(
                iterations   = trial.suggest_int  ("iterations",   it_lo, it_hi),
                learning_rate= trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True),
                depth        = trial.suggest_int  ("depth",        dep_lo, dep_hi),
                l2_leaf_reg  = trial.suggest_float("l2_leaf_reg",  l2_lo, l2_hi, log=True),
                random_seed  = config["model"]["catboost_params"]["random_seed"]
            )
            model_instance.fit(X_train_s, Y_train_s,
                               eval_set=(X_val_s, Y_val_s),
                               early_stopping_rounds=es_rounds,
                               use_best_model=True,
                               verbose=False)
            pred_train, pred_val = model_instance.predict(X_train_s), model_instance.predict(X_val_s)

        elif model_type == "XGB":
            base = config["optuna"]["xgb_params"]
            n_lo,n_hi   = _safe_int_range  (base,"n_estimators_min","n_estimators_max")
            lr_lo,lr_hi = _safe_float_range(base,"learning_rate_min","learning_rate_max")
            d_lo,d_hi   = _safe_int_range  (base,"max_depth_min","max_depth_max")
            a_lo,a_hi   = _safe_float_range(base,"reg_alpha_min","reg_alpha_max")
            l_lo,l_hi   = _safe_float_range(base,"reg_lambda_min","reg_lambda_max")
            es_rounds = config["model"]["xgb_params"].get("early_stopping_rounds", 50)

            model_instance = XGBRegression(
                n_estimators = trial.suggest_int  ("n_estimators", n_lo, n_hi),
                learning_rate= trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True),
                max_depth    = trial.suggest_int  ("max_depth", d_lo, d_hi),
                reg_alpha    = trial.suggest_float("reg_alpha", a_lo, a_hi, log=True),
                reg_lambda   = trial.suggest_float("reg_lambda", l_lo, l_hi, log=True),
                random_state = config["model"]["xgb_params"]["random_seed"]
            )
            model_instance = train_sklearn_model(model_instance, X_train_s, Y_train_s,
                                                 X_val=X_val_s, Y_val=Y_val_s,
                                                 enable_early_stop=True, es_rounds=es_rounds)
            pred_train, pred_val = model_instance.predict(X_train_s), model_instance.predict(X_val_s)

        else:
            raise ValueError(f"Tuning for {model_type} not implemented.")

        # -------- metrics & objective  ---------------------------------------
        m_train = compute_regression_metrics(Y_train_s, pred_train)
        m_val   = compute_regression_metrics(Y_val_s,   pred_val)

        ratio  = m_val["MSE"] / max(m_train["MSE"], 1e-8)
        alpha  = float(config.get("optuna", {}).get("overfit_penalty_alpha", 1.0))
        obj    = m_val["MSE"] + alpha * ratio

        trial.set_user_attr("MSE_STD_VAL",  m_val["MSE"])
        trial.set_user_attr("MSE_STD_TRAIN",m_train["MSE"])
        trial.set_user_attr("Penalty",      alpha * ratio)

        # 反标准化 R²
        trial.set_user_attr("R2_RAW",
            compute_regression_metrics(Y_val,
                inverse_transform_output(pred_val, sy))["R2"])

        return obj

    # 3) run Optuna -----------------------------------------------------------
    sampler = optuna.samplers.TPESampler(seed=random_seed)  # ★新增
    study = optuna.create_study(direction="minimize",  # ★改动
                                sampler=sampler)
    study.optimize(objective, n_trials=config["optuna"]["trials"])

    best_params = study.best_params
    print(f"[{model_type}] Best Obj={study.best_value:.6f}, params={best_params}")

    csv_name = os.path.splitext(os.path.basename(config["data"]["path"]))[0]
    optuna_dir = os.path.join("postprocessing", csv_name, "optuna", model_type)
    ensure_dir(optuna_dir)
    # study.trials_dataframe().to_csv(os.path.join(optuna_dir, "trials.csv"), index=False)

    return study, best_params


def create_model_by_type(model_type, config, random_seed=42, input_dim=None):
    csv_name = os.path.splitext(os.path.basename(config["data"]["path"]))[0]
    best_params = None
    optuna_dir = os.path.join("postprocessing", csv_name, "optuna", model_type)
    best_params_path = os.path.join(optuna_dir, "best_params.pkl")
    if os.path.exists(best_params_path):
        best_params = joblib.load(best_params_path)
        # 如果 hidden_dims 是字符串，则转换为 tuple
        if best_params and "hidden_dims" in best_params:
            if isinstance(best_params["hidden_dims"], str):
                best_params["hidden_dims"] = tuple(int(x.strip()) for x in best_params["hidden_dims"].split(','))
        print(f"[INFO] Using tuned hyperparameters for {model_type}: {best_params}")

    if model_type == "ANN":
        # 从 config 中取出 ann_params，并用 optuna 的 best_params 更新
        ann_cfg = config["model"].get("ann_params", {}).copy()
        ckpt_path = os.path.join(get_model_dir(csv_name, "ANN"), "best_ann.pt")
        ensure_dir(os.path.dirname(ckpt_path))  # <<< 新增，确保目录存在
        ann_cfg.setdefault("checkpoint_path",ckpt_path)
        if best_params:
            ann_cfg.update(best_params)
        # --- 新增两行 ---------------------------------------------------
        out_dim = config["data"]["output_len"]  # 读 yaml 里真正的输出列数
        ann_cfg["output_dim"] = out_dim  # 保存在字典里，后面训练要用
        # --------------------------------------------------------------
        actual_dim = input_dim if input_dim is not None else ann_cfg.get("input_dim", 14)
        model = ANNRegression(
            input_dim=actual_dim,
            # output_dim=ann_cfg.get("output_dim", 4),
            output_dim=out_dim,
            hidden_dims=ann_cfg["hidden_dims"],
            dropout=ann_cfg.get("dropout", 0.0),
            activation=ann_cfg.get("activation", "ReLU"),
            random_seed=ann_cfg.get("random_seed", random_seed)
        )
        # 返回模型和更新后的超参数字典
        return model, ann_cfg
    elif model_type == "RF":
        rf_cfg = config["model"]["rf_params"].copy()
        if best_params:
            rf_cfg.update(best_params)
        return RFRegression(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            random_state=rf_cfg["random_state"],
            ccp_alpha=rf_cfg.get("ccp_alpha", 0.0),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 1)
        )
    elif model_type == "DT":
        dt_cfg = config["model"]["dt_params"].copy()
        if best_params:
            dt_cfg.update(best_params)
        return DTRegression(
            max_depth=dt_cfg["max_depth"],
            random_state=dt_cfg["random_state"],
            ccp_alpha=dt_cfg.get("ccp_alpha", 0.0)
        )
    elif model_type == "CatBoost":
        cat_cfg = config["model"]["catboost_params"].copy()
        if best_params:
            cat_cfg.update(best_params)
        return CatBoostRegression(
            iterations=cat_cfg["iterations"],
            learning_rate=cat_cfg["learning_rate"],
            depth=cat_cfg["depth"],
            random_seed=cat_cfg["random_seed"],
            l2_leaf_reg=cat_cfg.get("l2_leaf_reg", 3.0)
        )
    elif model_type == "XGB":
        xgb_cfg = config["model"]["xgb_params"].copy()
        if best_params:
            xgb_cfg.update(best_params)
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




def train_main():
    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    csv_path = config["data"]["path"]
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    # 👇 这一行是新增
    root_model_dir = get_root_model_dir(csv_name)
    ensure_dir(root_model_dir)

    base_outdir = os.path.join("postprocessing", csv_name, "train")
    ensure_dir(base_outdir)

    # 1) 加载数据
    in_len = config["data"]["input_len"]
    out_len = config["data"]["output_len"]
    (X, Y, numeric_cols_idx, x_col_names, y_col_names,
     observed_combos, onehot_groups, oh_index_map) = load_dataset(csv_path, in_len, out_len)

    # 1.1) 保存 X_onehot.npy
    np.save(os.path.join(base_outdir, "X_onehot.npy"), X)
    np.save(os.path.join(base_outdir, "x_onehot_colnames.npy"), x_col_names)

    # 1.2) 若要做 raw correlation
    df_raw_14 = load_raw_data_for_correlation(csv_path, drop_nan=True, input_len=in_len, output_len=out_len, fill_same_as_train=True)
    raw_csv_path = os.path.join(base_outdir, "df_raw_14.csv")
    df_raw_14.to_csv(raw_csv_path, index=False)
    print(f"[INFO] Saved raw 14-col CSV => {raw_csv_path}")

    # 2) 提取统计信息 => metadata.pkl
    stats_dict = extract_data_statistics(X, x_col_names, numeric_cols_idx, Y=Y, y_col_names=y_col_names)
    stats_dict["numeric_cols_idx"] = numeric_cols_idx  # ← 这行新增
    stats_dict["onehot_groups"] = onehot_groups
    stats_dict["oh_index_map"]  = oh_index_map
    stats_dict["observed_onehot_combos"] = observed_combos
    stats_dict["x_col_names"] = x_col_names
    stats_dict["y_col_names"] = y_col_names

    meta_path = os.path.join(root_model_dir, "metadata.pkl")
    joblib.dump(stats_dict, meta_path)
    print(f"[INFO] metadata saved => {meta_path}")

    # 3) 数据拆分 & 标准化
    random_seed = config["data"].get("random_seed", 42)
    X_train, X_val, Y_train, Y_val = split_data(X, Y, test_size=config["data"]["test_size"], random_state=random_seed)
    bounded_cols = config["preprocessing"].get("bounded_output_columns", None)
    if bounded_cols is not None:
        bounded_indices = []
        for col in bounded_cols:
            if col in y_col_names:
                bounded_indices.append(y_col_names.index(col))
            else:
                print(f"[WARN] {col} not found in y_col_names")
    else:
        bounded_indices = None

    (X_train_s, X_val_s, sx), (Y_train_s, Y_val_s, sy) = standardize_data(
        X_train, X_val, Y_train, Y_val,
        do_input=config["preprocessing"]["standardize_input"],
        do_output=config["preprocessing"]["standardize_output"],
        numeric_cols_idx=numeric_cols_idx,
        do_output_bounded=(bounded_indices is not None) or config["preprocessing"].get("bounded_output", False),
        bounded_output_cols_idx=bounded_indices
    )

    np.save(os.path.join(base_outdir, "Y_train.npy"), Y_train)
    np.save(os.path.join(base_outdir, "Y_val.npy"), Y_val)

    # 4) 进行 Optuna 调参，并保存 best_params
    if config["optuna"].get("enable", False):
        for mtype in config["optuna"]["models"]:
            print(f"\n[INFO] Tuning hyperparameters for {mtype} ...")
            study, best_params = tune_model(mtype, config, X, Y, numeric_cols_idx, x_col_names, y_col_names, random_seed)
            optuna_dir = os.path.join("postprocessing", csv_name, "optuna", mtype)
            ensure_dir(optuna_dir)
            joblib.dump(study, os.path.join(optuna_dir, "study.pkl"))
            joblib.dump(best_params, os.path.join(optuna_dir, "best_params.pkl"))
            print(f"[INFO] Best params for {mtype} saved => {optuna_dir}")

    # 5) K‑Fold 交叉验证 (保存每折明细 + 过拟合度量)
    # --------------------------------------------------------------
    if config["evaluation"].get("do_cross_validation", False):
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        cv_metrics = {}  # <- 将写入 postprocessing/…/train/cv_metrics.pkl

        for mtype in config["model"]["types"]:
            print(f"[INFO] Running 5‑fold CV for model: {mtype}")

            # —— 每折分别记录 ——  (先建空 list)
            mse_tr, mse_va = [], []
            mae_tr, mae_va = [], []
            r2_tr, r2_va = [], []

            fold_id = 1
            for train_idx, val_idx in kf.split(X):
                print(f"  • Fold {fold_id}: train={len(train_idx)}, val={len(val_idx)}")
                # -------------------------------- split / scale
                X_tr, X_va = X[train_idx], X[val_idx]
                Y_tr, Y_va = Y[train_idx], Y[val_idx]

                (X_tr_s, X_va_s, _), (Y_tr_s, Y_va_s, sy_fold) = standardize_data(
                    X_tr, X_va, Y_tr, Y_va,
                    do_input=config["preprocessing"]["standardize_input"],
                    do_output=config["preprocessing"]["standardize_output"],
                    numeric_cols_idx=numeric_cols_idx,
                    do_output_bounded=(bounded_indices is not None) or
                                      config["preprocessing"].get("bounded_output", False),
                    bounded_output_cols_idx=bounded_indices
                )

                # -------------------------------- build & train model_cv
                if mtype == "ANN":
                    model_cv, ann_cfg = create_model_by_type(
                        "ANN", config, random_seed, input_dim=X_tr_s.shape[1]
                    )
                    if "epochs" not in ann_cfg:
                        ann_cfg["epochs"] = config["model"].get("ann_params", {}).get("epochs", 6000)

                    loss_fn = get_torch_loss_fn(config["loss"]["type"])
                    train_ds = MyDataset(X_tr_s, Y_tr_s)
                    val_ds = MyDataset(X_va_s, Y_va_s)

                    model_cv, _, _ = train_torch_model_dataloader(
                        model_cv, train_ds, val_ds,
                        loss_fn=loss_fn,
                        epochs=ann_cfg["epochs"],
                        batch_size=ann_cfg["batch_size"],
                        lr=float(ann_cfg["learning_rate"]),
                        weight_decay=float(ann_cfg.get("weight_decay", 0.0)),
                        checkpoint_path=None,
                        log_interval=config["training"]["log_interval"],
                        early_stopping=ann_cfg.get("early_stopping", True),
                        patience=ann_cfg.get("patience", 5),
                        optimizer_name=ann_cfg.get("optimizer", "Adam")
                    )
                    model_cv.eval().to("cpu")
                else:
                    model_cv = create_model_by_type(mtype, config, random_seed,
                                                    input_dim=X_tr_s.shape[1])
                    es_flag = mtype in ["CatBoost", "XGB"]
                    es_round = config["model"][f"{mtype.lower()}_params"].get(
                        "early_stopping_rounds", 50)
                    model_cv = train_sklearn_model(
                        model_cv, X_tr_s, Y_tr_s,
                        X_val=X_va_s, Y_val=Y_va_s,
                        enable_early_stop=es_flag, es_rounds=es_round
                    )

                # -------------------------------- 预测 (STD 域)
                if hasattr(model_cv, "eval") and hasattr(model_cv, "forward"):
                    with torch.no_grad():
                        pred_tr = model_cv(torch.tensor(X_tr_s, dtype=torch.float32)).cpu().numpy()
                        pred_va = model_cv(torch.tensor(X_va_s, dtype=torch.float32)).cpu().numpy()
                else:
                    pred_tr = model_cv.predict(X_tr_s)
                    pred_va = model_cv.predict(X_va_s)

                # -------------------------------- 评估 (STD 域 + 反标准化 R²)
                m_tr_std = compute_regression_metrics(Y_tr_s, pred_tr)
                m_va_std = compute_regression_metrics(Y_va_s, pred_va)

                # 存 list
                mse_tr.append(m_tr_std["MSE"]);
                mse_va.append(m_va_std["MSE"])
                mae_tr.append(m_tr_std["MAE"]);
                mae_va.append(m_va_std["MAE"])
                r2_tr.append(m_tr_std["R2"]);
                r2_va.append(m_va_std["R2"])

                print(f"    ↳ Fold‑{fold_id}  MSE={m_va_std['MSE']:.4f}  "
                      f"MAE={m_va_std['MAE']:.4f}  R²={m_va_std['R2']:.4f}")
                fold_id += 1

            # -------- 过拟合指标 per‑fold --------
            mse_ratio = [v / t if t != 0 else np.inf for v, t in zip(mse_va, mse_tr)]
            r2_diff = [t - v for v, t in zip(r2_va, r2_tr)]

            # -------- 汇总写入 dict --------
            cv_metrics[mtype] = {
                # ★旧字段：平均性能（给原先条形 + 雷达图继续用）
                "MSE": float(np.mean(mse_va)),
                "MAE": float(np.mean(mae_va)),
                "R2": float(np.mean(r2_va)),
                # ★新字段：明细 + 过拟合
                "folds": {
                    "MSE_train": mse_tr, "MSE_val": mse_va,
                    "MAE_train": mae_tr, "MAE_val": mae_va,
                    "R2_train": r2_tr, "R2_val": r2_va,
                    "MSE_ratio": mse_ratio,  # >1 越大越过拟合
                    "R2_diff": r2_diff  # >0 越大越过拟合
                }
            }
            print(f"[INFO] Finished 5‑fold CV for {mtype}")

        # —— 保存 ——  （路径保持不变，visualization 仍能找到）
        cv_metrics_path = os.path.join(base_outdir, "cv_metrics.pkl")
        joblib.dump(cv_metrics, cv_metrics_path)
        print(f"[INFO] 5‑fold CV metrics (detail) saved → {cv_metrics_path}")

    # 6) 正式训练 & 保存模型
    model_types = config["model"]["types"]
    metrics_rows = []  # ← Excel 行缓冲区
    for mtype in model_types:
        print(f"\n=== Train model: {mtype} ===")
        outdir_m = os.path.join(base_outdir, mtype)
        ensure_dir(outdir_m)
        # 针对ANN，解包返回的超参数字典
        if mtype == "ANN":
            model, ann_cfg = create_model_by_type(mtype, config, random_seed, input_dim=X_train_s.shape[1])
            if "epochs" not in ann_cfg:
                ann_cfg["epochs"] = config["model"].get("ann_params",{}).get("epochs", 6000)
            loss_fn = get_torch_loss_fn(config["loss"]["type"])
            train_ds = MyDataset(X_train_s, Y_train_s)
            val_ds = MyDataset(X_val_s, Y_val_s)
            # 正式训练阶段若存在checkpoint，则加载；否则按optuna初始化训练
            if os.path.exists(ann_cfg["checkpoint_path"]):
                print(f"[INFO] Found checkpoint for {mtype}, loading weights from {ann_cfg['checkpoint_path']}")
                # 示例代码： model.load_state_dict(torch.load(ann_cfg["checkpoint_path"]))
            model, train_losses, val_losses = train_torch_model_dataloader(
                model, train_ds, val_ds,
                loss_fn=loss_fn,
                epochs=ann_cfg["epochs"],
                batch_size=ann_cfg["batch_size"],
                lr=float(ann_cfg["learning_rate"]),
                weight_decay=float(ann_cfg.get("weight_decay", 0.0)),
                checkpoint_path=ann_cfg["checkpoint_path"],
                log_interval=config["training"]["log_interval"],
                early_stopping=ann_cfg.get("early_stopping", True),
                patience=ann_cfg.get("patience", 5),
                optimizer_name=ann_cfg.get("optimizer", "AdamW")
            )
            model.eval()
            model.to("cpu")
            np.save(os.path.join(outdir_m, "train_losses.npy"), train_losses)
            np.save(os.path.join(outdir_m, "val_losses.npy"), val_losses)
        else:
            model = create_model_by_type(mtype, config, random_seed, input_dim=X_train_s.shape[1])
            early = mtype in ["CatBoost", "XGB"]
            es_rounds = config["model"][f"{mtype.lower()}_params"].get("early_stopping_rounds", 50)
            model = train_sklearn_model(
                model, X_train_s, Y_train_s,
                X_val=X_val_s, Y_val=Y_val_s,
                enable_early_stop=early, es_rounds=es_rounds
            )
        # ---------- 预测（标准化域） ----------
        if hasattr(model, 'eval') and hasattr(model, 'forward'):
            with torch.no_grad():
                train_pred_std = model(torch.tensor(X_train_s, dtype=torch.float32)).cpu().numpy()
                val_pred_std = model(torch.tensor(X_val_s, dtype=torch.float32)).cpu().numpy()
        else:
            train_pred_std = model.predict(X_train_s)
            val_pred_std = model.predict(X_val_s)
        # ① 先把 std 结果统一成 2-D --------------------------
        train_pred_std = _to_2d(train_pred_std)
        val_pred_std = _to_2d(val_pred_std)
        # ------------------------------------------------------------------------

        # ---------- 反标准化 ----------
        train_pred_raw = inverse_transform_output(train_pred_std, sy) \
            if config["preprocessing"]["standardize_output"] else train_pred_std
        val_pred_raw = inverse_transform_output(val_pred_std, sy) \
            if config["preprocessing"]["standardize_output"] else val_pred_std
        # ② 再把 raw 结果也统一成 2-D ------------------------
        train_pred_raw = _to_2d(train_pred_raw)
        val_pred_raw = _to_2d(val_pred_raw)
        # ---------------------------------------------------
        # --- 把 y 也保证成 2-D（只需做一次） -------------------------
        Y_train_s = _to_2d(Y_train_s)
        Y_val_s = _to_2d(Y_val_s)
        Y_train = _to_2d(Y_train)
        Y_val = _to_2d(Y_val)
        # -------------------------------------------------------------

        # ---------- 计算三套指标 ----------
        std_tr = compute_regression_metrics(Y_train_s, train_pred_std)
        std_va = compute_regression_metrics(Y_val_s, val_pred_std)

        raw_tr = compute_regression_metrics(Y_train, train_pred_raw)
        raw_va = compute_regression_metrics(Y_val, val_pred_raw)

        mix_tr = {"MSE": std_tr["MSE"], "MAE": std_tr["MAE"], "R2": raw_tr["R2"]}
        mix_va = {"MSE": std_va["MSE"], "MAE": std_va["MAE"], "R2": raw_va["R2"]}

        print(f"   => train MIX={mix_tr},  valid MIX={mix_va}")

        # ---------- 保存 ----------
        np.save(os.path.join(outdir_m, "train_pred_std.npy"), train_pred_std)
        np.save(os.path.join(outdir_m, "val_pred_std.npy"), val_pred_std)
        np.save(os.path.join(outdir_m, "train_pred_raw.npy"), train_pred_raw)
        np.save(os.path.join(outdir_m, "val_pred_raw.npy"), val_pred_raw)

        joblib.dump(
            {
                "mixed": {"train": mix_tr, "val": mix_va},  # 主要查这个
                "std": {"train": std_tr, "val": std_va},
                "raw": {"train": raw_tr, "val": raw_va}
            },
            os.path.join(outdir_m, "metrics.pkl")
        )

        # ===========  (A) 组装一行  ===========
        out_names = y_col_names or [f"output{i}" for i in range(train_pred_raw.shape[1])]
        row_dict = {"model": mtype}

        for d, name in enumerate(out_names):
            # 1) R² —— RAW 域
            r2_tr = r2_score(Y_train[:, d], train_pred_raw[:, d])
            r2_va = r2_score(Y_val[:, d], val_pred_raw[:, d])

            # 2) MSE / MAE —— STD 域
            mse_tr = mean_squared_error(Y_train_s[:, d], train_pred_std[:, d])
            mse_va = mean_squared_error(Y_val_s[:, d], val_pred_std[:, d])

            mae_tr = mean_absolute_error(Y_train_s[:, d], train_pred_std[:, d])
            mae_va = mean_absolute_error(Y_val_s[:, d], val_pred_std[:, d])

            # 3) 写进 dict
            row_dict[("train", name, "r2")] = r2_tr
            row_dict[("train", name, "mse")] = mse_tr
            row_dict[("train", name, "mae")] = mae_tr
            row_dict[("valid", name, "r2")] = r2_va
            row_dict[("valid", name, "mse")] = mse_va
            row_dict[("valid", name, "mae")] = mae_va
        # ---------------------------------------------------------------------
        metrics_rows.append(row_dict)
        # ===========  (A) 结束  ===========

        ### <<< PATCH: save to per‑data model_dir ###
        model_dir = get_model_dir(csv_name, mtype)
        ensure_dir(model_dir)

        if mtype == "ANN":
            torch.save(model.state_dict(),
                       os.path.join(model_dir, "best_ann.pt"))
            ann_meta = {
                "input_dim": X_train.shape[1],
                "best_val_loss": float(min(val_losses)),
                "epoch": int(np.argmin(val_losses))
            }
            with open(os.path.join(model_dir, "ann_meta.json"), "w") as f:
                json.dump(ann_meta, f, indent=2)
        else:
            joblib.dump(model,
                        os.path.join(model_dir, "trained_model.pkl"))

        save_scaler(sx, os.path.join(model_dir, f"scaler_x_{mtype}.pkl"))
        save_scaler(sy, os.path.join(model_dir, f"scaler_y_{mtype}.pkl"))

        np.save(os.path.join(model_dir, "x_col_names.npy"),
                np.array(x_col_names, dtype=object))
        np.save(os.path.join(model_dir, "y_col_names.npy"),
                np.array(y_col_names, dtype=object))

        #shap
        if config["evaluation"].get("save_shap", False):
            shap_dir = os.path.join("evaluation/figures", csv_name, "model_comparison", mtype, "shap")
            ensure_dir(shap_dir)
            X_full = np.concatenate([X_train, X_val], axis=0)
            X_full_s = np.concatenate([X_train_s, X_val_s], axis=0)
            try:
                if mtype == "ANN":
                    model.eval()
                    background = torch.tensor(X_train_s[:100], dtype=torch.float32)
                    explainer = shap.DeepExplainer(model, background)
                    shap_values = explainer.shap_values(torch.tensor(X_full_s, dtype=torch.float32))
                elif mtype == "CatBoost":
                    shap_values = model.get_shap_values(X_full_s)
                else:
                    base_model = model.model if hasattr(model, "model") else model
                    explainer = shap.TreeExplainer(base_model)
                    shap_values = explainer.shap_values(X_full_s)
                shap_save = {
                    "shap_values": shap_values,
                    "X_full": X_full_s,
                    "x_col_names": x_col_names,
                    "y_col_names": y_col_names
                }
                shap_save_path = os.path.join(shap_dir, "shap_data.pkl")
                joblib.dump(shap_save, shap_save_path)
                print(f"[INFO] SHAP data saved for model {mtype} => {shap_save_path}")
            except Exception as e:
                print(f"[WARN] SHAP computation failed for {mtype}: {e}")
        print("\n[INFO] train_main => done.")
    # ------------------------------------------------------------------
    #  (B) 汇总写 Excel  —— 训练循环结束以后一次性写出
    # ------------------------------------------------------------------
    if metrics_rows:  # 防止 metrics_rows 为空
        # === 1) 组装 DataFrame =================================================
        df = pd.DataFrame(metrics_rows)

        # === 2) 把三层列名拍平成一层，并做「安全化」 ============================
        #   规则：1) 去掉空格 / %, () / 斜线等特殊符号 → 下划线
        #         2) 多个连续下划线合并成 1 个
        #         3) 去掉首尾下划线
        def _safe_name(text: str) -> str:
            text = str(text).strip()
            text = re.sub(r"[^\w]", "_", text)  # 非字母数字下划线 → _
            text = re.sub(r"_+", "_", text)  # 折叠多余下划线
            return text.strip("_")

        flat_cols = []
        for col in df.columns:
            # 原列是 tuple 三层：(stage, outputName, metric)；首列 "model" 是 str
            if isinstance(col, tuple):
                flat_cols.append("_".join(_safe_name(c) for c in col if c))
            else:
                flat_cols.append(_safe_name(col))
        df.columns = flat_cols

        # === 3) 写 Excel ========================================================
        dest_dir = os.path.join("evaluation", "figures", csv_name)
        ensure_dir(dest_dir)
        excel_path = os.path.join(dest_dir, "metrics_summary.xlsx")

        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="metrics")

            # 可选：简单调列宽，首列 12，其余 18
            worksheet = writer.sheets["metrics"]
            for i in range(len(df.columns)):
                worksheet.set_column(i, i, 12 if i == 0 else 18)

        print(f"[INFO] Excel metrics summary saved → {excel_path}")
    # ------------------------------------------------------------------


if __name__ == "__main__":
    train_main()
