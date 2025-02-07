"""
main.py

Project entry point:
1. Load config
2. Load dataset, split data
3. Standardize data (optional)
4. Train model (ANN, RF, GAN, DT, CatBoost, XGB)
5. Evaluate and visualize
6. Print final metrics
"""

import yaml
import numpy as np
import torch

from data_preprocessing.data_loader import load_dataset, load_raw_data_for_correlation
from data_preprocessing.data_split import split_data
from data_preprocessing.my_dataset import MyDataset
from data_preprocessing.scaler_utils import (
    standardize_data, save_scaler, load_scaler, inverse_transform_output
)

from models.model_ann import ANNRegression
from models.model_rf import RFRegression
from models.model_gan import GANModel
# 新增导入:
from models.model_dt import DTRegression
from models.model_catboost import CatBoostRegression
from models.model_xgb import XGBRegression

from losses.torch_losses import get_torch_loss_fn

from trainers.train_torch import train_torch_model_dataloader
from trainers.train_sklearn import train_sklearn_model

from evaluation.metrics import compute_regression_metrics
from evaluation.visualization import (
    plot_loss_curve,
    plot_scatter_3d_outputs_mse,
    plot_scatter_3d_outputs_mae,
    plot_residual_histogram,
    plot_residual_kde,
    plot_correlation_heatmap,
    plot_rf_feature_importance_bar,
    plot_2d_heatmap
)

def main():
    # 1. Load config
    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Load data
    X, Y, numeric_cols_idx, x_col_names, y_col_names = load_dataset(config["data"]["path"])

    # 2.1 相关性分析可视化 (原始14列 & One-Hot 后)
    if config["evaluation"].get("save_correlation", False):
        df_for_corr = load_raw_data_for_correlation(config["data"]["path"], drop_nan=True)
        col_names = df_for_corr.columns.tolist()

        non_hot_numeric_cols = df_for_corr.select_dtypes(include=[np.number]).columns.tolist()
        non_hot_numeric_cols_idx = [col_names.index(c) for c in non_hot_numeric_cols]

        # 未 One-Hot 的相关性热力图
        plot_correlation_heatmap(
            X=df_for_corr.values,
            col_names=col_names,
            numeric_cols_idx=non_hot_numeric_cols_idx,
            filename="correlation_heatmap.jpg",
            method_numeric="pearson"
        )
        # One-Hot 后的相关性热力图
        plot_correlation_heatmap(
            X,
            col_names=x_col_names,
            numeric_cols_idx=range(X.shape[1]),
            filename="correlation_heatmap_one_hot.jpg",
            method_numeric="pearson"
        )

    # 3. Split data
    X_train, X_val, Y_train, Y_val = split_data(
        X, Y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_seed"]
    )

    # 4. Standardize
    (X_train_s, X_val_s, scaler_x), (Y_train_s, Y_val_s, scaler_y) = standardize_data(
        X_train, X_val, Y_train, Y_val,
        do_input=config["preprocessing"]["standardize_input"],
        do_output=config["preprocessing"]["standardize_output"],
        numeric_cols_idx=numeric_cols_idx
    )

    print("X_train_s any NaN?", np.isnan(X_train_s).any())
    print("X_train_s any Inf?", np.isinf(X_train_s).any())

    save_scaler(scaler_x, "./scaler_x.pkl")
    if config["preprocessing"]["standardize_output"]:
        save_scaler(scaler_y, "./scaler_y.pkl")

    # 5. Choose model
    model_type = config["model"]["type"]
    if model_type in ["ANN", "GAN"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # 5.1 Train model
    if model_type == "ANN":
        ann_cfg = config["model"]["ann_params"]
        input_dim = X_train_s.shape[1]
        output_dim = ann_cfg["output_dim"]
        model = ANNRegression(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=ann_cfg["hidden_dims"]
        ).to(device)

        loss_fn = get_torch_loss_fn(config["loss"]["type"])
        train_dataset = MyDataset(X_train_s, Y_train_s)
        val_dataset   = MyDataset(X_val_s, Y_val_s)
        model, train_losses, val_losses = train_torch_model_dataloader(
            model,
            train_dataset,
            val_dataset,
            loss_fn=loss_fn,
            epochs=ann_cfg["epochs"],
            batch_size=ann_cfg["batch_size"],
            lr=float(ann_cfg["learning_rate"]),
            checkpoint_path=ann_cfg["checkpoint_path"],
            log_interval=config["training"]["log_interval"]
        )
        if config["evaluation"]["save_loss_curve"]:
            plot_loss_curve(train_losses, val_losses, filename="ann_loss_curve.jpg")

        # Predict
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(X_val_s, dtype=torch.float32, device=device)).cpu().numpy()

    elif model_type == "RF":
        rf_cfg = config["model"]["rf_params"]
        model = RFRegression(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            random_state=config["data"]["random_seed"]
        )
        model = train_sklearn_model(model, X_train_s, Y_train_s)
        val_pred = model.predict(X_val_s)

        if config["evaluation"].get("save_feature_importance_bar", False):
            plot_rf_feature_importance_bar(
                rf_model=model.model,
                feature_names=x_col_names,
                filename="rf_feature_importance.jpg",
                top_k=20
            )

    elif model_type == "GAN":
        print("[INFO] Using placeholder GANModel.")
        model = GANModel()
        model.fit(X_train_s, Y_train_s)
        val_pred = model.predict(X_val_s)

    # ---------------- NEW MODELS ----------------
    elif model_type == "DT":
        dt_cfg = config["model"]["dt_params"]
        model = DTRegression(
            max_depth=dt_cfg["max_depth"],
            random_state=dt_cfg["random_state"]
        )
        model = train_sklearn_model(model, X_train_s, Y_train_s)
        val_pred = model.predict(X_val_s)

        # 决策树也有 feature_importances_
        if config["evaluation"].get("save_feature_importance_bar", False):
            plot_rf_feature_importance_bar(
                rf_model=model.model,
                feature_names=x_col_names,
                filename="dt_feature_importance.jpg",
                top_k=20
            )

    elif model_type == "CatBoost":
        cat_cfg = config["model"]["catboost_params"]
        model = CatBoostRegression(
            iterations=cat_cfg["iterations"],
            learning_rate=cat_cfg["learning_rate"],
            depth=cat_cfg["depth"],
            random_seed=cat_cfg["random_seed"]
        )
        model = train_sklearn_model(model, X_train_s, Y_train_s)
        val_pred = model.predict(X_val_s)

        if config["evaluation"].get("save_feature_importance_bar", False):
            plot_rf_feature_importance_bar(
                rf_model=model.model,
                feature_names=x_col_names,
                filename="catboost_feature_importance.jpg",
                top_k=20
            )

    elif model_type == "XGB":
        xgb_cfg = config["model"]["xgb_params"]
        model = XGBRegression(
            n_estimators=xgb_cfg["n_estimators"],
            learning_rate=xgb_cfg["learning_rate"],
            max_depth=xgb_cfg["max_depth"],
            random_state=xgb_cfg["random_seed"]
        )
        model = train_sklearn_model(model, X_train_s, Y_train_s)
        val_pred = model.predict(X_val_s)

        if config["evaluation"].get("save_feature_importance_bar", False):
            plot_rf_feature_importance_bar(
                rf_model=model.model,
                feature_names=x_col_names,
                filename="xgb_feature_importance.jpg",
                top_k=20
            )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 6. Inverse transform predictions if needed
    if config["preprocessing"]["standardize_output"]:
        val_pred = inverse_transform_output(val_pred, scaler_y)

    # 7. Visualization
    if config["evaluation"]["save_scatter_mse_plot"]:
        plot_scatter_3d_outputs_mse(Y_val, val_pred, y_labels=y_col_names, filename=f"{model_type}_mse_scatter.jpg")
    if config["evaluation"]["save_scatter_mae_plot"]:
        plot_scatter_3d_outputs_mae(Y_val, val_pred, y_labels=y_col_names, filename=f"{model_type}_mae_scatter.jpg")
    if config["evaluation"]["save_residual_hist"]:
        plot_residual_histogram(Y_val, val_pred, y_labels=y_col_names, filename=f"{model_type}_residual_hist.jpg")
    if config["evaluation"]["save_residual_kde"]:
        plot_residual_kde(Y_val, val_pred, y_labels=y_col_names, filename=f"{model_type}_residual_kde.jpg")

    if config["evaluation"]["save_heatmap"]:
        # 例: 针对 2 numeric 列扫描
        # 需要包装一下 model 以提供 .predict() 接口
        if model_type == "ANN":
            class ANNWrapper:
                def __init__(self, net, scaler_y):
                    self.net = net
                    self.scaler_y = scaler_y
                    self.device = next(net.parameters()).device
                def predict(self, X):
                    X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        out = self.net(X_t).cpu().numpy()
                    if self.scaler_y is not None:
                        out = inverse_transform_output(out, self.scaler_y)
                    return out
            model_for_heatmap = ANNWrapper(model, scaler_y)
        else:
            class GenericWrapper:
                def __init__(self, m, sy):
                    self.m = m
                    self.sy = sy
                def predict(self, X):
                    preds = self.m.predict(X)
                    if self.sy is not None:
                        preds = inverse_transform_output(preds, self.sy)
                    return preds
            model_for_heatmap = GenericWrapper(model, scaler_y)

        # num_features = X_train_s.shape[1]
        # if len(numeric_cols_idx) >= 2:
        #     col_x = numeric_cols_idx[0]
        #     col_y = numeric_cols_idx[1]
        #     base_input = np.zeros(num_features, dtype=float)
        #     plot_2d_heatmap(
        #         model_for_heatmap,
        #         x_range=(0, 1),
        #         y_range=(0, 1),
        #         resolution=30,
        #         num_features=num_features,
        #         col_x=col_x,
        #         col_y=col_y,
        #         fixed_input=base_input,
        #         input_scaler=scaler_x,
        #         output_scaler=None,
        #         output_index=0,
        #         filename=f"{model_type}_heatmap_out0.jpg"
        #     )

    # 8. Compute metrics
    metrics_res = compute_regression_metrics(Y_val, val_pred)
    print(f"{model_type} Validation Metrics:", metrics_res)

    # 9. Simple inference for 1 sample
    sample_x = X_val_s[:1]
    if model_type == "ANN":
        with torch.no_grad():
            sample_pred = model(torch.tensor(sample_x, dtype=torch.float32, device=device)).cpu().numpy()
    elif model_type in ["RF", "GAN", "DT", "CatBoost", "XGB"]:
        sample_pred = model.predict(sample_x)
    else:
        sample_pred = None

    if config["preprocessing"]["standardize_output"] and sample_pred is not None:
        sample_pred = inverse_transform_output(sample_pred, scaler_y)

    print(f"Sample inference => input={X_val[0]}, pred={sample_pred[0] if sample_pred is not None else None}, true={Y_val[0]}")

if __name__ == "__main__":
    main()
