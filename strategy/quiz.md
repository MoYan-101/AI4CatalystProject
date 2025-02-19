我的代码项目如下图所示 

❯ tree   
.
├── AI4CatalystProject
│   ├── catboost_info
│   │   ├── catboost_training.json
│   │   ├── learn
│   │   │   └── events.out.tfevents
│   │   ├── learn_error.tsv
│   │   ├── time_left.tsv
│   │   └── tmp
│   ├── configs
│   │   └── config.yaml
│   ├── data
│   │   ├── cleansing.ipynb
│   │   ├── Dataset_20250203.csv
│   │   ├── Dataset_20250203_upd.csv
│   │   ├── Dataset_20250203_upd_without1.csv
│   │   ├── Dataset_20250203_upd.xlsx
│   │   ├── Dataset_20250203.xlsx
│   │   ├── Dataset_20250205.csv
│   │   ├── Dataset_20250205_final.csv
│   │   ├── Dataset_20250205_without1.csv
│   │   ├── Dataset_20250210_Cat.csv
│   │   ├── Dataset_20250210_Cat.xlsx
│   │   ├── Dataset_20250210_log.csv
│   │   ├── Dataset_20250210_log.xlsx
│   │   ├── Dataset_20250214_final.csv
│   │   ├── dataset.csv
│   │   ├── jupyter.log
│   │   └── pycharm.log
│   ├── data_preprocessing
│   │   ├── data_loader.py
│   │   ├── data_split.py
│   │   ├── __init__.py
│   │   ├── my_dataset.py
│   │   ├── __pycache__
│   │   │   ├── data_loader.cpython-38.pyc
│   │   │   ├── data_split.cpython-38.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── my_dataset.cpython-38.pyc
│   │   │   └── scaler_utils.cpython-38.pyc
│   │   └── scaler_utils.py
│   ├── evaluation
│   │   ├── figures
│   │   │   ├── Archive.zip
│   │   │   ├── category
│   │   │   │   ├── DataCorrelation_log
│   │   │   │   │   ├── correlation_heatmap.jpg
│   │   │   │   │   ├── correlation_heatmap_one_hot.jpg
│   │   │   │   │   └── three_metrics_horizontal.jpg
│   │   │   │   └── model_comparison_log
│   │   │   │       ├── CatBoost
│   │   │   │       │   ├── catboost_feature_importance.jpg
│   │   │   │       │   ├── CatBoost_mae_scatter.jpg
│   │   │   │       │   ├── CatBoost_mse_scatter.jpg
│   │   │   │       │   ├── CatBoost_residual_hist.jpg
│   │   │   │       │   └── CatBoost_residual_kde.jpg
│   │   │   │       ├── DT
│   │   │   │       │   ├── dt_feature_importance.jpg
│   │   │   │       │   ├── DT_mae_scatter.jpg
│   │   │   │       │   ├── DT_mse_scatter.jpg
│   │   │   │       │   ├── DT_residual_hist.jpg
│   │   │   │       │   └── DT_residual_kde.jpg
│   │   │   │       ├── RF
│   │   │   │       │   ├── rf_feature_importance.jpg
│   │   │   │       │   ├── RF_mae_scatter.jpg
│   │   │   │       │   ├── RF_mse_scatter.jpg
│   │   │   │       │   ├── RF_residual_hist.jpg
│   │   │   │       │   └── RF_residual_kde.jpg
│   │   │   │       └── XGB
│   │   │   │           ├── xgb_feature_importance.jpg
│   │   │   │           ├── XGB_mae_scatter.jpg
│   │   │   │           ├── XGB_mse_scatter.jpg
│   │   │   │           ├── XGB_residual_hist.jpg
│   │   │   │           └── XGB_residual_kde.jpg
│   │   │   ├── correlation_heatmap.jpg
│   │   │   ├── correlation_heatmap_one_hot.jpg
│   │   │   ├── dt_feature_importance.jpg
│   │   │   ├── DT_mae_scatter.jpg
│   │   │   ├── DT_mse_scatter.jpg
│   │   │   ├── DT_residual_hist.jpg
│   │   │   ├── DT_residual_kde.jpg
│   │   │   ├── log
│   │   │   │   ├── DataCorrelation_log
│   │   │   │   │   ├── correlation_heatmap.jpg
│   │   │   │   │   ├── correlation_heatmap_one_hot.jpg
│   │   │   │   │   └── three_metrics_horizontal.jpg
│   │   │   │   └── model_comparison_log
│   │   │   │       ├── CatBoost
│   │   │   │       │   ├── catboost_feature_importance.jpg
│   │   │   │       │   ├── CatBoost_mae_scatter.jpg
│   │   │   │       │   ├── CatBoost_mse_scatter.jpg
│   │   │   │       │   ├── CatBoost_residual_hist.jpg
│   │   │   │       │   └── CatBoost_residual_kde.jpg
│   │   │   │       ├── DT
│   │   │   │       │   ├── dt_feature_importance.jpg
│   │   │   │       │   ├── DT_mae_scatter.jpg
│   │   │   │       │   ├── DT_mse_scatter.jpg
│   │   │   │       │   ├── DT_residual_hist.jpg
│   │   │   │       │   └── DT_residual_kde.jpg
│   │   │   │       ├── RF
│   │   │   │       │   ├── rf_feature_importance.jpg
│   │   │   │       │   ├── RF_mae_scatter.jpg
│   │   │   │       │   ├── RF_mse_scatter.jpg
│   │   │   │       │   ├── RF_residual_hist.jpg
│   │   │   │       │   └── RF_residual_kde.jpg
│   │   │   │       └── XGB
│   │   │   │           ├── xgb_feature_importance.jpg
│   │   │   │           ├── XGB_mae_scatter.jpg
│   │   │   │           ├── XGB_mse_scatter.jpg
│   │   │   │           ├── XGB_residual_hist.jpg
│   │   │   │           └── XGB_residual_kde.jpg
│   │   │   └── normal
│   │   │       ├── DataCorrelation
│   │   │       │   ├── correlation_heatmap.jpg
│   │   │       │   ├── correlation_heatmap_one_hot.jpg
│   │   │       │   └── three_metrics_horizontal.jpg
│   │   │       └── model_comparison
│   │   │           ├── CatBoost
│   │   │           │   ├── catboost_feature_importance.jpg
│   │   │           │   ├── CatBoost_mae_scatter.jpg
│   │   │           │   ├── CatBoost_mse_scatter.jpg
│   │   │           │   ├── CatBoost_residual_hist.jpg
│   │   │           │   └── CatBoost_residual_kde.jpg
│   │   │           ├── DT
│   │   │           │   ├── dt_feature_importance.jpg
│   │   │           │   ├── DT_mae_scatter.jpg
│   │   │           │   ├── DT_mse_scatter.jpg
│   │   │           │   ├── DT_residual_hist.jpg
│   │   │           │   └── DT_residual_kde.jpg
│   │   │           ├── RF
│   │   │           │   ├── rf_feature_importance.jpg
│   │   │           │   ├── RF_mae_scatter.jpg
│   │   │           │   ├── RF_mse_scatter.jpg
│   │   │           │   ├── RF_residual_hist.jpg
│   │   │           │   └── RF_residual_kde.jpg
│   │   │           └── XGB
│   │   │               ├── xgb_feature_importance.jpg
│   │   │               ├── XGB_mae_scatter.jpg
│   │   │               ├── XGB_mse_scatter.jpg
│   │   │               ├── XGB_residual_hist.jpg
│   │   │               └── XGB_residual_kde.jpg
│   │   ├── figures.zip
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── metrics.cpython-38.pyc
│   │   │   └── visualization.cpython-38.pyc
│   │   └── visualization.py
│   ├── losses
│   │   ├── __init__.py
│   │   ├── placeholder_loss.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── torch_losses.cpython-38.pyc
│   │   └── torch_losses.py
│   ├── main.py
│   ├── models
│   │   ├── best_ann.pt
│   │   ├── __init__.py
│   │   ├── model_ann.py
│   │   ├── model_catboost.py
│   │   ├── model_dt.py
│   │   ├── model_gan.py
│   │   ├── model_rf.py
│   │   ├── model_xgb.py
│   │   └── __pycache__
│   │       ├── __init__.cpython-38.pyc
│   │       ├── model_ann.cpython-38.pyc
│   │       ├── model_catboost.cpython-38.pyc
│   │       ├── model_dt.cpython-38.pyc
│   │       ├── model_gan.cpython-38.pyc
│   │       ├── model_rf.cpython-38.pyc
│   │       └── model_xgb.cpython-38.pyc
│   ├── performance.py
│   ├── __pycache__
│   │   └── main.cpython-38-pytest-7.1.2.pyc
│   ├── README.md
│   ├── requirements.txt
│   ├── scaler_x.pkl
│   └── trainers
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-38.pyc
│       │   ├── train_sklearn.cpython-38.pyc
│       │   └── train_torch.cpython-38.pyc
│       ├── train_sklearn.py
│       └── train_torch.py
├── KFAI4CatalystProject
│   ├── catboost_info
│   │   ├── catboost_training.json
│   │   ├── learn
│   │   │   └── events.out.tfevents
│   │   ├── learn_error.tsv
│   │   ├── time_left.tsv
│   │   └── tmp
│   ├── check.md
│   ├── configs
│   │   └── config.yaml
│   ├── data
│   │   ├── cleansing.ipynb
│   │   ├── Dataset_20250203.csv
│   │   ├── Dataset_20250203_upd.csv
│   │   ├── Dataset_20250203_upd_without1.csv
│   │   ├── Dataset_20250203_upd.xlsx
│   │   ├── Dataset_20250203.xlsx
│   │   ├── Dataset_20250205.csv
│   │   ├── Dataset_20250205_final.csv
│   │   ├── Dataset_20250205_without1.csv
│   │   ├── Dataset_20250210_Cat.csv
│   │   ├── Dataset_20250210_Cat.xlsx
│   │   ├── Dataset_20250210_log.csv
│   │   ├── Dataset_20250210_log.xlsx
│   │   ├── dataset.csv
│   │   ├── jupyter.log
│   │   └── pycharm.log
│   ├── data_preprocessing
│   │   ├── data_loader.py
│   │   ├── data_split.py
│   │   ├── __init__.py
│   │   ├── my_dataset.py
│   │   ├── __pycache__
│   │   │   ├── data_loader.cpython-38.pyc
│   │   │   ├── data_split.cpython-38.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── my_dataset.cpython-38.pyc
│   │   │   └── scaler_utils.cpython-38.pyc
│   │   └── scaler_utils.py
│   ├── Dataset_20250205_final
│   │   ├── DataCorrelation
│   │   └── model_comparison
│   │       ├── RF
│   │       └── XGB
│   ├── evaluation
│   │   ├── figures
│   │   │   ├── Dataset_20250205_final
│   │   │   │   ├── DataCorrelation
│   │   │   │   │   ├── catalyst_size_vs_product.jpg
│   │   │   │   │   ├── correlation_heatmap.jpg
│   │   │   │   │   ├── correlation_heatmap_one_hot.jpg
│   │   │   │   │   ├── kde_distribution.jpg
│   │   │   │   │   ├── overfitting_single.jpg
│   │   │   │   │   ├── potential_vs_product_by_electrolyte.jpg
│   │   │   │   │   ├── product_distribution.jpg
│   │   │   │   │   ├── three_metrics_horizontal_train.jpg
│   │   │   │   │   └── three_metrics_horizontal_val.jpg
│   │   │   │   └── model_comparison
│   │   │   │       ├── ANN
│   │   │   │       │   ├── ANN_loss_curve.jpg
│   │   │   │       │   └── full
│   │   │   │       │       ├── train
│   │   │   │       │       │   ├── ANN_mae_scatter_train.jpg
│   │   │   │       │       │   ├── ANN_mse_scatter_train.jpg
│   │   │   │       │       │   ├── ANN_residual_hist_train.jpg
│   │   │   │       │       │   └── ANN_residual_kde_train.jpg
│   │   │   │       │       └── valid
│   │   │   │       │           ├── ANN_mae_scatter.jpg
│   │   │   │       │           ├── ANN_mse_scatter.jpg
│   │   │   │       │           ├── ANN_residual_hist.jpg
│   │   │   │       │           └── ANN_residual_kde.jpg
│   │   │   │       ├── CatBoost
│   │   │   │       │   ├── CatBoost_feature_importance.jpg
│   │   │   │       │   └── full
│   │   │   │       │       ├── train
│   │   │   │       │       │   ├── CatBoost_mae_scatter_train.jpg
│   │   │   │       │       │   ├── CatBoost_mse_scatter_train.jpg
│   │   │   │       │       │   ├── CatBoost_residual_hist_train.jpg
│   │   │   │       │       │   └── CatBoost_residual_kde_train.jpg
│   │   │   │       │       └── valid
│   │   │   │       │           ├── CatBoost_mae_scatter.jpg
│   │   │   │       │           ├── CatBoost_mse_scatter.jpg
│   │   │   │       │           ├── CatBoost_residual_hist.jpg
│   │   │   │       │           └── CatBoost_residual_kde.jpg
│   │   │   │       ├── DT
│   │   │   │       │   ├── DT_feature_importance.jpg
│   │   │   │       │   └── full
│   │   │   │       │       ├── train
│   │   │   │       │       │   ├── DT_mae_scatter_train.jpg
│   │   │   │       │       │   ├── DT_mse_scatter_train.jpg
│   │   │   │       │       │   ├── DT_residual_hist_train.jpg
│   │   │   │       │       │   └── DT_residual_kde_train.jpg
│   │   │   │       │       └── valid
│   │   │   │       │           ├── DT_mae_scatter.jpg
│   │   │   │       │           ├── DT_mse_scatter.jpg
│   │   │   │       │           ├── DT_residual_hist.jpg
│   │   │   │       │           └── DT_residual_kde.jpg
│   │   │   │       └── RF
│   │   │   │           ├── full
│   │   │   │           │   ├── train
│   │   │   │           │   │   ├── RF_mae_scatter_train.jpg
│   │   │   │           │   │   ├── RF_mse_scatter_train.jpg
│   │   │   │           │   │   ├── RF_residual_hist_train.jpg
│   │   │   │           │   │   └── RF_residual_kde_train.jpg
│   │   │   │           │   └── valid
│   │   │   │           │       ├── RF_mae_scatter.jpg
│   │   │   │           │       ├── RF_mse_scatter.jpg
│   │   │   │           │       ├── RF_residual_hist.jpg
│   │   │   │           │       └── RF_residual_kde.jpg
│   │   │   │           └── RF_feature_importance.jpg
│   │   │   ├── Dataset_20250205_final.zip
│   │   │   └── fig_20250211.tar.gz
│   │   ├── figures.zip
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── metrics.cpython-38.pyc
│   │   │   └── visualization.cpython-38.pyc
│   │   └── visualization.py
│   ├── inference.py
│   ├── losses
│   │   ├── __init__.py
│   │   ├── placeholder_loss.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── torch_losses.cpython-38.pyc
│   │   └── torch_losses.py
│   ├── main.py
│   ├── models
│   │   ├── ANN
│   │   │   ├── best_ann.pt
│   │   │   ├── scaler_x_ANN.pkl
│   │   │   ├── scaler_y_ANN.pkl
│   │   │   ├── x_col_names.npy
│   │   │   └── y_col_names.npy
│   │   ├── CatBoost
│   │   │   ├── scaler_x_CatBoost.pkl
│   │   │   ├── scaler_y_CatBoost.pkl
│   │   │   ├── trained_model.pkl
│   │   │   ├── x_col_names.npy
│   │   │   └── y_col_names.npy
│   │   ├── DT
│   │   │   ├── scaler_x_DT.pkl
│   │   │   ├── scaler_y_DT.pkl
│   │   │   ├── trained_model.pkl
│   │   │   ├── x_col_names.npy
│   │   │   └── y_col_names.npy
│   │   ├── __init__.py
│   │   ├── model_ann.py
│   │   ├── model_catboost.py
│   │   ├── model_dt.py
│   │   ├── model_gan.py
│   │   ├── model_rf.py
│   │   ├── model_xgb.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── model_ann.cpython-38.pyc
│   │   │   ├── model_catboost.cpython-38.pyc
│   │   │   ├── model_dt.cpython-38.pyc
│   │   │   ├── model_gan.cpython-38.pyc
│   │   │   ├── model_rf.cpython-38.pyc
│   │   │   └── model_xgb.cpython-38.pyc
│   │   └── RF
│   │       ├── scaler_x_RF.pkl
│   │       ├── scaler_y_RF.pkl
│   │       ├── trained_model.pkl
│   │       ├── x_col_names.npy
│   │       └── y_col_names.npy
│   ├── performance.py
│   ├── __pycache__
│   │   └── main.cpython-38-pytest-7.1.2.pyc
│   ├── README.md
│   ├── requirements.txt
│   └── trainers
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-38.pyc
│       │   ├── train_sklearn.cpython-38.pyc
│       │   └── train_torch.cpython-38.pyc
│       ├── train_sklearn.py
│       └── train_torch.py
├── mkdir.sh
├── NewAI4CatalystProject
│   ├── catboost_info
│   │   ├── catboost_training.json
│   │   ├── learn
│   │   │   └── events.out.tfevents
│   │   ├── learn_error.tsv
│   │   ├── time_left.tsv
│   │   └── tmp
│   ├── configs
│   │   └── config.yaml
│   ├── data
│   │   ├── cleansing.ipynb
│   │   ├── Dataset_20250203.csv
│   │   ├── Dataset_20250203_upd.csv
│   │   ├── Dataset_20250203_upd_without1.csv
│   │   ├── Dataset_20250203_upd.xlsx
│   │   ├── Dataset_20250203.xlsx
│   │   ├── Dataset_20250205.csv
│   │   ├── Dataset_20250205_final.csv
│   │   ├── Dataset_20250205_without1.csv
│   │   ├── Dataset_20250210_Cat.csv
│   │   ├── Dataset_20250210_Cat.xlsx
│   │   ├── Dataset_20250210_log.csv
│   │   ├── Dataset_20250210_log.xlsx
│   │   ├── dataset.csv
│   │   ├── jupyter.log
│   │   └── pycharm.log
│   ├── data_preprocessing
│   │   ├── data_loader.py
│   │   ├── data_split.py
│   │   ├── __init__.py
│   │   ├── my_dataset.py
│   │   ├── __pycache__
│   │   │   ├── data_loader.cpython-38.pyc
│   │   │   ├── data_split.cpython-38.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── my_dataset.cpython-38.pyc
│   │   │   └── scaler_utils.cpython-38.pyc
│   │   └── scaler_utils.py
│   ├── Dataset_20250205_final
│   │   ├── DataCorrelation
│   │   └── model_comparison
│   │       ├── RF
│   │       └── XGB
│   ├── evaluation
│   │   ├── figures
│   │   │   ├── Dataset_20250205_final
│   │   │   │   ├── DataCorrelation
│   │   │   │   │   ├── correlation_heatmap.jpg
│   │   │   │   │   ├── correlation_heatmap_one_hot.jpg
│   │   │   │   │   └── three_metrics_horizontal.jpg
│   │   │   │   └── model_comparison
│   │   │   │       ├── CatBoost
│   │   │   │       │   ├── CatBoost_feature_importance.jpg
│   │   │   │       │   └── full
│   │   │   │       │       ├── train
│   │   │   │       │       │   ├── CatBoost_mae_scatter_train.jpg
│   │   │   │       │       │   ├── CatBoost_mse_scatter_train.jpg
│   │   │   │       │       │   ├── CatBoost_residual_hist_train.jpg
│   │   │   │       │       │   └── CatBoost_residual_kde_train.jpg
│   │   │   │       │       └── valid
│   │   │   │       │           ├── CatBoost_mae_scatter.jpg
│   │   │   │       │           ├── CatBoost_mse_scatter.jpg
│   │   │   │       │           ├── CatBoost_residual_hist.jpg
│   │   │   │       │           └── CatBoost_residual_kde.jpg
│   │   │   │       ├── DT
│   │   │   │       │   ├── DT_feature_importance.jpg
│   │   │   │       │   └── full
│   │   │   │       │       ├── train
│   │   │   │       │       │   ├── DT_mae_scatter_train.jpg
│   │   │   │       │       │   ├── DT_mse_scatter_train.jpg
│   │   │   │       │       │   ├── DT_residual_hist_train.jpg
│   │   │   │       │       │   └── DT_residual_kde_train.jpg
│   │   │   │       │       └── valid
│   │   │   │       │           ├── DT_mae_scatter.jpg
│   │   │   │       │           ├── DT_mse_scatter.jpg
│   │   │   │       │           ├── DT_residual_hist.jpg
│   │   │   │       │           └── DT_residual_kde.jpg
│   │   │   │       └── RF
│   │   │   │           ├── full
│   │   │   │           │   ├── train
│   │   │   │           │   │   ├── RF_mae_scatter_train.jpg
│   │   │   │           │   │   ├── RF_mse_scatter_train.jpg
│   │   │   │           │   │   ├── RF_residual_hist_train.jpg
│   │   │   │           │   │   └── RF_residual_kde_train.jpg
│   │   │   │           │   └── valid
│   │   │   │           │       ├── RF_mae_scatter.jpg
│   │   │   │           │       ├── RF_mse_scatter.jpg
│   │   │   │           │       ├── RF_residual_hist.jpg
│   │   │   │           │       └── RF_residual_kde.jpg
│   │   │   │           └── RF_feature_importance.jpg
│   │   │   └── Dataset_20250205_final_without_xgb.zip
│   │   ├── figures.zip
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── metrics.cpython-38.pyc
│   │   │   └── visualization.cpython-38.pyc
│   │   └── visualization.py
│   ├── losses
│   │   ├── __init__.py
│   │   ├── placeholder_loss.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── torch_losses.cpython-38.pyc
│   │   └── torch_losses.py
│   ├── main.py
│   ├── models
│   │   ├── best_ann.pt
│   │   ├── __init__.py
│   │   ├── model_ann.py
│   │   ├── model_catboost.py
│   │   ├── model_dt.py
│   │   ├── model_gan.py
│   │   ├── model_rf.py
│   │   ├── model_xgb.py
│   │   └── __pycache__
│   │       ├── __init__.cpython-38.pyc
│   │       ├── model_ann.cpython-38.pyc
│   │       ├── model_catboost.cpython-38.pyc
│   │       ├── model_dt.cpython-38.pyc
│   │       ├── model_gan.cpython-38.pyc
│   │       ├── model_rf.cpython-38.pyc
│   │       └── model_xgb.cpython-38.pyc
│   ├── performance.py
│   ├── __pycache__
│   │   └── main.cpython-38-pytest-7.1.2.pyc
│   ├── README.md
│   ├── requirements.txt
│   ├── scaler_x.pkl
│   └── trainers
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-38.pyc
│       │   ├── train_sklearn.cpython-38.pyc
│       │   └── train_torch.cpython-38.pyc
│       ├── train_sklearn.py
│       └── train_torch.py
├── PLTKFAI4CatalystProject
│   ├── ANN_inference
│   │   ├── 2d_heatmap
│   │   │   ├── heatmap_output_1.png
│   │   │   ├── heatmap_output_2.png
│   │   │   ├── heatmap_output_3.png
│   │   │   └── heatmap_output_4.png
│   │   └── confusion_matrix
│   │       └── confusion_matrix_mimo.png
│   ├── catboost_info
│   │   ├── catboost_training.json
│   │   ├── learn
│   │   │   └── events.out.tfevents
│   │   ├── learn_error.tsv
│   │   ├── time_left.tsv
│   │   └── tmp
│   ├── check.md
│   ├── configs
│   │   └── config.yaml
│   ├── data
│   │   ├── cleansing.ipynb
│   │   ├── Dataset_20250203.csv
│   │   ├── Dataset_20250203_upd.csv
│   │   ├── Dataset_20250203_upd_without1.csv
│   │   ├── Dataset_20250203_upd.xlsx
│   │   ├── Dataset_20250203.xlsx
│   │   ├── Dataset_20250205.csv
│   │   ├── Dataset_20250205_final.csv
│   │   ├── Dataset_20250205_without1.csv
│   │   ├── Dataset_20250210_Cat.csv
│   │   ├── Dataset_20250210_Cat.xlsx
│   │   ├── Dataset_20250210_log.csv
│   │   ├── Dataset_20250210_log.xlsx
│   │   ├── Dataset_20250214_final_2.csv
│   │   ├── Dataset_20250214_final_3.csv
│   │   ├── Dataset_20250214_final.csv
│   │   ├── dataset.csv
│   │   ├── jupyter.log
│   │   └── pycharm.log
│   ├── data_preprocessing
│   │   ├── data_loader.py
│   │   ├── data_split.py
│   │   ├── __init__.py
│   │   ├── my_dataset.py
│   │   ├── __pycache__
│   │   │   ├── data_loader.cpython-38.pyc
│   │   │   ├── data_split.cpython-38.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── my_dataset.cpython-38.pyc
│   │   │   └── scaler_utils.cpython-38.pyc
│   │   └── scaler_utils.py
│   ├── evaluation
│   │   ├── figures
│   │   │   ├── Dataset_20250214_final
│   │   │   │   ├── DataCorrelation
│   │   │   │   │   ├── box_product_vs_potential_bin.jpg
│   │   │   │   │   ├── box_product_vs_shape.jpg
│   │   │   │   │   ├── catalyst_size_vs_product.jpg
│   │   │   │   │   ├── correlation_heatmap.jpg
│   │   │   │   │   ├── correlation_heatmap_one_hot.jpg
│   │   │   │   │   ├── kde_distribution.jpg
│   │   │   │   │   ├── overfitting_single.jpg
│   │   │   │   │   ├── potential_vs_product_by_electrolyte.jpg
│   │   │   │   │   ├── product_distribution.jpg
│   │   │   │   │   ├── three_dot_potential_vs_product.jpg
│   │   │   │   │   ├── three_metrics_horizontal_train.jpg
│   │   │   │   │   └── three_metrics_horizontal_val.jpg
│   │   │   │   ├── DataCorrelation.zip
│   │   │   │   └── model_comparison
│   │   │   │       ├── ANN
│   │   │   │       │   ├── ANN_loss_curve.jpg
│   │   │   │       │   └── full
│   │   │   │       │       ├── train
│   │   │   │       │       │   ├── ANN_mae_scatter_train.jpg
│   │   │   │       │       │   ├── ANN_mse_scatter_train.jpg
│   │   │   │       │       │   ├── ANN_residual_hist_train.jpg
│   │   │   │       │       │   └── ANN_residual_kde_train.jpg
│   │   │   │       │       └── valid
│   │   │   │       │           ├── ANN_mae_scatter.jpg
│   │   │   │       │           ├── ANN_mse_scatter.jpg
│   │   │   │       │           ├── ANN_residual_hist.jpg
│   │   │   │       │           └── ANN_residual_kde.jpg
│   │   │   │       ├── CatBoost
│   │   │   │       │   ├── CatBoost_feature_importance.jpg
│   │   │   │       │   └── full
│   │   │   │       │       ├── train
│   │   │   │       │       │   ├── CatBoost_mae_scatter_train.jpg
│   │   │   │       │       │   ├── CatBoost_mse_scatter_train.jpg
│   │   │   │       │       │   ├── CatBoost_residual_hist_train.jpg
│   │   │   │       │       │   └── CatBoost_residual_kde_train.jpg
│   │   │   │       │       └── valid
│   │   │   │       │           ├── CatBoost_mae_scatter.jpg
│   │   │   │       │           ├── CatBoost_mse_scatter.jpg
│   │   │   │       │           ├── CatBoost_residual_hist.jpg
│   │   │   │       │           └── CatBoost_residual_kde.jpg
│   │   │   │       ├── DT
│   │   │   │       │   ├── DT_feature_importance.jpg
│   │   │   │       │   └── full
│   │   │   │       │       ├── train
│   │   │   │       │       │   ├── DT_mae_scatter_train.jpg
│   │   │   │       │       │   ├── DT_mse_scatter_train.jpg
│   │   │   │       │       │   ├── DT_residual_hist_train.jpg
│   │   │   │       │       │   └── DT_residual_kde_train.jpg
│   │   │   │       │       └── valid
│   │   │   │       │           ├── DT_mae_scatter.jpg
│   │   │   │       │           ├── DT_mse_scatter.jpg
│   │   │   │       │           ├── DT_residual_hist.jpg
│   │   │   │       │           └── DT_residual_kde.jpg
│   │   │   │       └── RF
│   │   │   │           ├── full
│   │   │   │           │   ├── train
│   │   │   │           │   │   ├── RF_mae_scatter_train.jpg
│   │   │   │           │   │   ├── RF_mse_scatter_train.jpg
│   │   │   │           │   │   ├── RF_residual_hist_train.jpg
│   │   │   │           │   │   └── RF_residual_kde_train.jpg
│   │   │   │           │   └── valid
│   │   │   │           │       ├── RF_mae_scatter.jpg
│   │   │   │           │       ├── RF_mse_scatter.jpg
│   │   │   │           │       ├── RF_residual_hist.jpg
│   │   │   │           │       └── RF_residual_kde.jpg
│   │   │   │           └── RF_feature_importance.jpg
│   │   │   ├── Dataset_20250214_final_2
│   │   │   │   ├── DataCorrelation
│   │   │   │   └── model_comparison
│   │   │   └── Dataset_20250214_final_3
│   │   │       ├── DataCorrelation
│   │   │       │   ├── box_product_vs_shape.jpg
│   │   │       │   ├── catalyst_size_vs_product.jpg
│   │   │       │   ├── correlation_heatmap.jpg
│   │   │       │   ├── correlation_heatmap_one_hot.jpg
│   │   │       │   ├── kde_distribution.jpg
│   │   │       │   ├── overfitting_single.jpg
│   │   │       │   ├── potential_vs_product_by_electrolyte.jpg
│   │   │       │   ├── product_distribution.jpg
│   │   │       │   ├── three_metrics_horizontal_train.jpg
│   │   │       │   └── three_metrics_horizontal_val.jpg
│   │   │       └── model_comparison
│   │   │           ├── ANN
│   │   │           │   ├── ANN_loss_curve.jpg
│   │   │           │   └── full
│   │   │           │       ├── train
│   │   │           │       │   ├── ANN_mae_scatter_train.jpg
│   │   │           │       │   ├── ANN_mse_scatter_train.jpg
│   │   │           │       │   ├── ANN_residual_hist_train.jpg
│   │   │           │       │   └── ANN_residual_kde_train.jpg
│   │   │           │       └── valid
│   │   │           │           ├── ANN_mae_scatter.jpg
│   │   │           │           ├── ANN_mse_scatter.jpg
│   │   │           │           ├── ANN_residual_hist.jpg
│   │   │           │           └── ANN_residual_kde.jpg
│   │   │           ├── CatBoost
│   │   │           │   ├── CatBoost_feature_importance.jpg
│   │   │           │   └── full
│   │   │           │       ├── train
│   │   │           │       │   ├── CatBoost_mae_scatter_train.jpg
│   │   │           │       │   ├── CatBoost_mse_scatter_train.jpg
│   │   │           │       │   ├── CatBoost_residual_hist_train.jpg
│   │   │           │       │   └── CatBoost_residual_kde_train.jpg
│   │   │           │       └── valid
│   │   │           │           ├── CatBoost_mae_scatter.jpg
│   │   │           │           ├── CatBoost_mse_scatter.jpg
│   │   │           │           ├── CatBoost_residual_hist.jpg
│   │   │           │           └── CatBoost_residual_kde.jpg
│   │   │           ├── DT
│   │   │           │   ├── DT_feature_importance.jpg
│   │   │           │   └── full
│   │   │           │       ├── train
│   │   │           │       │   ├── DT_mae_scatter_train.jpg
│   │   │           │       │   ├── DT_mse_scatter_train.jpg
│   │   │           │       │   ├── DT_residual_hist_train.jpg
│   │   │           │       │   └── DT_residual_kde_train.jpg
│   │   │           │       └── valid
│   │   │           │           ├── DT_mae_scatter.jpg
│   │   │           │           ├── DT_mse_scatter.jpg
│   │   │           │           ├── DT_residual_hist.jpg
│   │   │           │           └── DT_residual_kde.jpg
│   │   │           └── RF
│   │   │               ├── full
│   │   │               │   ├── train
│   │   │               │   │   ├── RF_mae_scatter_train.jpg
│   │   │               │   │   ├── RF_mse_scatter_train.jpg
│   │   │               │   │   ├── RF_residual_hist_train.jpg
│   │   │               │   │   └── RF_residual_kde_train.jpg
│   │   │               │   └── valid
│   │   │               │       ├── RF_mae_scatter.jpg
│   │   │               │       ├── RF_mse_scatter.jpg
│   │   │               │       ├── RF_residual_hist.jpg
│   │   │               │       └── RF_residual_kde.jpg
│   │   │               └── RF_feature_importance.jpg
│   │   ├── figures.zip
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── metrics.cpython-38.pyc
│   │   │   └── visualization.cpython-38.pyc
│   │   └── visualization.py
│   ├── inference.py
│   ├── losses
│   │   ├── __init__.py
│   │   ├── placeholder_loss.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   └── torch_losses.cpython-38.pyc
│   │   └── torch_losses.py
│   ├── main.py
│   ├── models
│   │   ├── ANN
│   │   │   ├── best_ann.pt
│   │   │   ├── scaler_x_ANN.pkl
│   │   │   ├── scaler_y_ANN.pkl
│   │   │   ├── x_col_names.npy
│   │   │   └── y_col_names.npy
│   │   ├── CatBoost
│   │   │   ├── scaler_x_CatBoost.pkl
│   │   │   ├── scaler_y_CatBoost.pkl
│   │   │   ├── trained_model.pkl
│   │   │   ├── x_col_names.npy
│   │   │   └── y_col_names.npy
│   │   ├── DT
│   │   │   ├── scaler_x_DT.pkl
│   │   │   ├── scaler_y_DT.pkl
│   │   │   ├── trained_model.pkl
│   │   │   ├── x_col_names.npy
│   │   │   └── y_col_names.npy
│   │   ├── __init__.py
│   │   ├── metadata.pkl
│   │   ├── model_ann.py
│   │   ├── model_catboost.py
│   │   ├── model_dt.py
│   │   ├── model_gan.py
│   │   ├── model_rf.py
│   │   ├── model_xgb.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── model_ann.cpython-38.pyc
│   │   │   ├── model_catboost.cpython-38.pyc
│   │   │   ├── model_dt.cpython-38.pyc
│   │   │   ├── model_gan.cpython-38.pyc
│   │   │   ├── model_rf.cpython-38.pyc
│   │   │   └── model_xgb.cpython-38.pyc
│   │   └── RF
│   │       ├── scaler_x_RF.pkl
│   │       ├── scaler_y_RF.pkl
│   │       ├── trained_model.pkl
│   │       ├── x_col_names.npy
│   │       └── y_col_names.npy
│   ├── performance.py
│   ├── __pycache__
│   │   └── main.cpython-38-pytest-7.1.2.pyc
│   ├── README.md
│   ├── requirements.txt
│   ├── RF_inference
│   │   └── 2d_heatmap
│   └── trainers
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-38.pyc
│       │   ├── train_sklearn.cpython-38.pyc
│       │   └── train_torch.cpython-38.pyc
│       ├── train_sklearn.py
│       └── train_torch.py
└── ValPLTKFAI4CatalystProject
    ├── ANN_inference
    │   ├── 2d_heatmap
    │   │   ├── heatmap_output_1.png
    │   │   ├── heatmap_output_2.png
    │   │   ├── heatmap_output_3.png
    │   │   └── heatmap_output_4.png
    │   └── confusion_matrix
    │       └── confusion_matrix_mimo.png
    ├── CatBoost_inference
    │   ├── 2d_heatmap
    │   │   ├── heatmap_output_1.png
    │   │   ├── heatmap_output_2.png
    │   │   ├── heatmap_output_3.png
    │   │   └── heatmap_output_4.png
    │   └── confusion_matrix
    │       └── confusion_matrix_mimo.png
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
    │   ├── Dataset_20250214_final_2.csv
    │   ├── Dataset_20250214_final_3.csv
    │   ├── Dataset_20250214_final.csv
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
    ├── evaluation
    │   ├── figures
    │   │   ├── Dataset_20250214_final_3
    │   │   │   ├── DataCorrelation
    │   │   │   │   ├── box_product_vs_catalyst.jpg
    │   │   │   │   ├── box_product_vs_potential_bin.jpg
    │   │   │   │   ├── box_product_vs_shape.jpg
    │   │   │   │   ├── catalyst_size_vs_product.jpg
    │   │   │   │   ├── correlation_heatmap.jpg
    │   │   │   │   ├── correlation_heatmap_one_hot.jpg
    │   │   │   │   ├── kde_distribution.jpg
    │   │   │   │   ├── overfitting_single.jpg
    │   │   │   │   ├── potential_vs_product_by_electrolyte.jpg
    │   │   │   │   ├── product_distribution.jpg
    │   │   │   │   ├── three_dot_potential_vs_product.jpg
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
    │   │   └── Dataset_20250214_final_3.zip
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
    ├── model_parameters.md
    ├── model_parameters.pdf
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
    │   ├── metadata.pkl
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

239 directories, 737 files

下面开始逐个展示我的代码:
配置文件:

```
data:
#  path: "./data/Dataset_20250205_without1.csv"   # CSV file path
  path: "./data/Dataset_20250214_final_3.csv"   # CSV file path
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
  bounded_output: true

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
    patience: 120                 # 允许更多的耐心等待
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
  models: [ "ANN", "CatBoost"]   # 例如只想推理这两个
  max_combinations: 20           # 若分类组合数 > 20, 则截断简化
  # 你也可以加 output_dir, 但下面逻辑里我们自动放在 <csv_name>/inference/<model_type> 下
```

数据读取:

```
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

★ 不包含任何 logit transform. 只保留你原先的逻辑.
"""

import pandas as pd
import numpy as np

def load_dataset(csv_path, drop_nan=True):
    """
    Read a CSV file, forcibly keep only the first 14 columns.
    Then parse the first 10 as X, the next 4 as Y.
    Perform one-hot encoding for categorical features in X.
    Finally, return X (NumPy), Y (NumPy), numeric_cols_idx, x_col_names, y_col_names.
    """

    # 1) 读取CSV, 强制只保留前14列
    df_raw = pd.read_csv(csv_path)
    df = df_raw.iloc[:, :14].copy()

    # 2) 处理空值
    if drop_nan:
        df.dropna(subset=df.columns, how='any', inplace=True)

    # 3) 拆分 X_df, Y_df
    if df.shape[1] < 14:
        raise ValueError("After dropping/cleaning, not enough columns remain (need 14).")

    X_df = df.iloc[:, :10].copy()
    Y_df = df.iloc[:, 10:14].copy()  # shape: (N,4)
    y_col_names = list(Y_df.columns)

    Y = Y_df.values

    # (可选)检查 Y 是否有负值或>100
    for i, cname in enumerate(y_col_names):
        below0 = (Y[:,i] < 0).sum()
        above100 = (Y[:,i] > 100).sum()
        if below0>0 or above100>0:
            print(f"[WARN] Column '{cname}' => {below0} vals<0, {above100} vals>100 total.")

    # 4) 分析 X_df
    categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols_original = [col for col in X_df.columns if col not in categorical_cols]

    # 5) One-hot
    X_encoded = pd.get_dummies(X_df, columns=categorical_cols)
    all_cols = X_encoded.columns.tolist()

    numeric_cols_idx = []
    for i, colname in enumerate(all_cols):
        if colname in numeric_cols_original:
            numeric_cols_idx.append(i)

    X = X_encoded.values
    x_col_names = list(X_encoded.columns)

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


def extract_data_statistics(X, x_col_names, numeric_cols_idx):
    """
    提取并返回一个 dict:
      {
        "continuous_cols": {
            "colname": {
                "min": ...,
                "max": ...,
                "mean": ...
            }, ...
        },
        "onehot_groups": [
            ...
        ]
      }
    用于推理时做网格/枚举.
    """
    stats = {
        "continuous_cols": {},
        "onehot_groups": []
    }

    # 1) 连续列信息
    for idx in numeric_cols_idx:
        cname = x_col_names[idx]
        col_data = X[:, idx]
        stats["continuous_cols"][cname] = {
            "min": float(np.min(col_data)),
            "max": float(np.max(col_data)),
            "mean": float(np.mean(col_data))
        }

    # 2) one-hot分组
    prefix_map = {}
    for i, cname in enumerate(x_col_names):
        if i not in numeric_cols_idx:
            # 说明这是 one-hot 列
            if '_' in cname:
                prefix = cname.split('_')[0]
            else:
                prefix = cname
            prefix_map.setdefault(prefix, []).append(i)

    for pref, idxs in prefix_map.items():
        if len(idxs)>=2:
            stats["onehot_groups"].append(sorted(idxs))
        else:
            stats["onehot_groups"].append([idxs[0]])

    return stats
```

数据集合分解:

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

深度学习dataset:

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

数据mapping标准化+物理意义mapping(0-100):

```
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
    """
    y in [0,100] => z in [-1,1], linear:
        z = 2*(y/100) - 1
    clamp y to [0,100] just in case
    """
    y_ = np.clip(y, 0, 100)
    return 2*(y_/100.0) - 1

def inverse_bounded_transform(z):
    """
    z => clamp in [-1,1], => y=100*(z+1)/2 in [0,100]
    """
    z_ = np.clip(z, -1, 1)
    return 100.0*(z_+1.0)/2.0

def standardize_data(X_train, X_val,
                     Y_train, Y_val,
                     do_input=True,
                     do_output=False,
                     numeric_cols_idx=None,
                     do_output_bounded=False):
    """
    Optionally standardize input features (X) and/or output targets (Y).
    If do_output_bounded=True => 0..100 => -1..1 => standard => model.
    """
    scaler_x = None
    scaler_y = None

    X_train_s = np.copy(X_train)
    X_val_s   = np.copy(X_val)
    Y_train_s = np.copy(Y_train)
    Y_val_s   = np.copy(Y_val)

    if do_input:
        if numeric_cols_idx is None:
            numeric_cols_idx = list(range(X_train.shape[1]))
        scaler_x = StandardScaler()
        scaler_x.fit(X_train_s[:, numeric_cols_idx])
        X_train_s[:, numeric_cols_idx] = scaler_x.transform(X_train_s[:, numeric_cols_idx])
        X_val_s[:, numeric_cols_idx]   = scaler_x.transform(X_val_s[:, numeric_cols_idx])

    if do_output:
        # bounded or not
        if do_output_bounded:
            # 0..100 => -1..1
            for i in range(Y_train_s.shape[1]):
                Y_train_s[:,i] = bounded_transform(Y_train_s[:,i])
                Y_val_s[:,i]   = bounded_transform(Y_val_s[:,i])
            transform_type = "bounded+standard"
        else:
            transform_type = "standard"

        # standard
        scaler_obj = StandardScaler()
        scaler_obj.fit(Y_train_s)
        Y_train_s = scaler_obj.transform(Y_train_s)
        Y_val_s   = scaler_obj.transform(Y_val_s)

        scaler_y = {
            "type": transform_type,
            "scaler": scaler_obj
        }

    return (X_train_s, X_val_s, scaler_x), (Y_train_s, Y_val_s, scaler_y)

def save_scaler(scaler, path):
    if scaler is not None:
        joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)

def inverse_transform_output(y_pred, scaler_y):
    """
    If scaler_y["type"]=="bounded+standard": inverse standard => inverse_bounded => [0,100].
    If "standard": just inverse standard.
    """
    if scaler_y is None:
        return y_pred
    if not isinstance(scaler_y, dict):
        # older usage => direct standard
        return scaler_y.inverse_transform(y_pred)

    transform_type = scaler_y["type"]
    scaler_obj = scaler_y["scaler"]
    # 1) inverse standard
    y_ = scaler_obj.inverse_transform(y_pred)

    if transform_type.startswith("bounded"):
        # each col => clamp => [0,100]
        for i in range(y_.shape[1]):
            y_[:,i] = inverse_bounded_transform(y_[:,i])

    return y_
```

matric:

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

绘图可视化lib:

```
"""
evaluation/visualization.py

Contains functions for plotting:
- Loss curve (train/val)
- Scatter plot (True vs. Pred) with color = mse/mae
- Residual histograms / KDE
- Mixed correlation heatmap
- Data analysis plots (KDE, scatter, box, etc.)
- Feature importance
- Horizontal bar of metrics comparison
- Overfitting bar
- 推理分析可视化 (2D Heatmap + Confusion-like)
[本版在 plot_advanced_data_analysis 中新增了:
   1) boxplot(产物 vs Potential_bin)
   2) boxplot(产物 vs Shape)
   3) boxplot(产物 vs Catalyst)
   4) plot_potential_vs_product (仅对 Potential vs. 产物做三点图)
]
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

# 全局字体
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['axes.labelsize'] = 13
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titlesize'] = 15
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 12


def ensure_fig_dir(dir_path=FIG_DIR):
    """
    Ensure that the directory 'dir_path' exists.
    If not provided, default is FIG_DIR = './evaluation/figures'.
    """
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

# ------------------------------
# 对数值-数值, 类别-类别, 数值-类别的相关性
# ------------------------------
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    phi2 = max(0, chi2 - (k - 1)*(r - 1)/(n-1))
    r_adj = r - (r-1)**2/(n-1)
    k_adj = k - (k-1)**2/(n-1)
    denom = min(k_adj-1, r_adj-1)
    if denom == 0:
        return 0.0
    else:
        return math.sqrt(phi2/denom)

def correlation_ratio(categories, values):
    df = pd.DataFrame({'cat': categories, 'val': values})
    df.dropna(inplace=True)
    group_means = df.groupby('cat')['val'].mean()
    mean_all = df['val'].mean()

    ss_between = 0
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
    """
    n_features = X.shape[1]
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=col_names)

    numeric_cols_set = set(numeric_cols_idx)
    corr_matrix = np.zeros((n_features, n_features), dtype=float)
    used_method_matrix = [["" for _ in range(n_features)] for _ in range(n_features)]

    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                corr_matrix[i, j] = 1.0
                used_method_matrix[i][j] = "self"
                continue
            if i > j:
                corr_matrix[i, j] = corr_matrix[j, i]
                used_method_matrix[i][j] = used_method_matrix[j][i]
                continue

            col_i = col_names[i]
            col_j = col_names[j]
            data_i = X[col_i]
            data_j = X[col_j]
            if dropna:
                valid_mask = ~data_i.isna() & ~data_j.isna()
                data_i = data_i[valid_mask]
                data_j = data_j[valid_mask]

            i_is_num = (i in numeric_cols_set)
            j_is_num = (j in numeric_cols_set)

            if i_is_num and j_is_num:
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
                # 类别-类别
                r = cramers_v(data_i, data_j)
                corr_matrix[i, j] = r
                used_method_matrix[i][j] = "cramers_v"
            else:
                # 数值-类别
                if i_is_num:
                    cat_data, num_data = data_j, data_i
                else:
                    cat_data, num_data = data_i, data_j
                r = correlation_ratio(cat_data, num_data)
                corr_matrix[i, j] = r
                used_method_matrix[i][j] = "corr_ratio"

    for i in range(n_features):
        for j in range(i):
            corr_matrix[i, j] = corr_matrix[j, i]
            used_method_matrix[i][j] = used_method_matrix[j][i]

    return corr_matrix, used_method_matrix


def plot_correlation_heatmap(X,
                             col_names,
                             numeric_cols_idx,
                             filename="correlation_heatmap.jpg",
                             method_numeric="pearson",
                             cmap="ocean",
                             vmin=-1,
                             vmax=1):
    """
    绘制混合变量相关性热力图，并保存。
    """
    ensure_fig_dir()
    corr_matrix, used_methods = mixed_correlation_matrix(
        X, col_names, numeric_cols_idx,
        method_numeric=method_numeric
    )

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(col_names)), max(8, 0.5 * len(col_names))))

    sns.heatmap(corr_matrix,
                xticklabels=col_names,
                yticklabels=col_names,
                cmap=cmap,
                annot=False,
                square=True,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
                cbar_kws={"shrink": 0.8, "aspect": 30, "label": "Correlation"})

    ax.set_title("Mixed Correlation Heatmap", fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=700)
    plt.close()
    print(f"[plot_correlation_heatmap] Saved => {save_path}")


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
      - 在标题处显示该维度的 R² 值.
    """
    ensure_fig_dir()
    if y_pred.ndim != 2:
        raise ValueError(f"y_pred shape expected (N, n_outputs), got {y_pred.shape}")
    _, n_outputs = y_pred.shape

    fig, axes = plt.subplots(1, n_outputs, figsize=(4 * n_outputs, 4), squeeze=False)

    for i in range(n_outputs):
        errors = (y_true[:, i] - y_pred[:, i]) ** 2
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
    ensure_fig_dir()
    if y_pred.ndim != 2:
        raise ValueError(f"y_pred shape expected (N, n_outputs), got {y_pred.shape}")
    _, n_outputs = y_pred.shape

    fig, axes = plt.subplots(1, n_outputs, figsize=(4 * n_outputs, 4), squeeze=False)

    for i in range(n_outputs):
        errors = np.abs(y_true[:, i] - y_pred[:, i])
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
    ensure_fig_dir()
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


def plot_rf_feature_importance_bar(
        rf_model,
        feature_names,
        filename="rf_feature_importance_bar.jpg",
        top_k=20,
        threshold=0.05
):
    """
    使用柱状图绘制随机森林(或类似)特征重要度
    """
    ensure_fig_dir()
    importances = rf_model.feature_importances_
    if importances is None or len(importances) == 0:
        print("[WARNING] No importances found or importances is empty!")
        return

    sorted_indices = np.argsort(importances)[::-1]
    topk_indices = sorted_indices[:top_k]

    topk_features = [feature_names[i] for i in topk_indices]
    topk_importances = importances[topk_indices]
    colors = ["red" if imp > threshold else "blue" for imp in topk_importances]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(range(len(topk_importances)), topk_importances, align="center", color=colors)

    ax.set_yticks(range(len(topk_importances)))
    ax.set_yticklabels(topk_features, fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Feature Importance (Top-{top_k})", fontsize=14, fontweight='bold')

    ax.axvspan(0, threshold, facecolor='lightgray', alpha=0.5)
    ax.axvline(x=threshold, color='red', linestyle='dashed', linewidth=2)

    legend_elements = [
        Patch(facecolor="red", label=f"Importance > {threshold}"),
        Patch(facecolor="royalblue", label=f"Importance ≤ {threshold}")
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path, dpi=700)
    plt.close()

    print(f"[plot_rf_feature_importance_bar] Saved => {save_path}")


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
    ax.barh(y_positions, arr, color=colors, alpha=0.8, height=width)
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

    plot_hbar_with_mean(
        ax=axes[0],
        model_names=model_names,
        values=mse_vals,
        subplot_label="(a)",
        metric_label="MSE (Lower is better)",
        bigger_is_better=False,
        width=0.4
    )
    plot_hbar_with_mean(
        ax=axes[1],
        model_names=model_names,
        values=mae_vals,
        subplot_label="(b)",
        metric_label="MAE (Lower is better)",
        bigger_is_better=False,
        width=0.4
    )
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


from matplotlib.patches import Patch

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
    ax.barh(y_positions, arr, color=colors, alpha=0.8, height=width)
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


def plot_overfitting_horizontal(
    overfit_data,
    save_name="overfitting_horizontal.jpg"
):
    ensure_fig_dir()
    model_names = list(overfit_data.keys())

    msr_vals = [overfit_data[m]["MSE_ratio"] for m in model_names]
    r2d_vals = [overfit_data[m]["R2_diff"]   for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_hbar_with_threshold(
        ax=axes[0],
        model_names=model_names,
        values=msr_vals,
        subplot_label="(a)",
        metric_label="MSE Ratio (Val/Train)\n(Lower is better)",
        bigger_is_better=False,
        width=0.4,
        threshold_h=10
    )
    plot_hbar_with_threshold(
        ax=axes[1],
        model_names=model_names,
        values=r2d_vals,
        subplot_label="(b)",
        metric_label="R2 diff (Train - Val)\n(Lower is better)",
        bigger_is_better=False,
        width=0.4,
        threshold_h=0.2,
        threshold_l=0.15
    )

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, save_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot_overfitting_horizontal] => {out_path}")


# ========== 数据分析可视化 ==========

def plot_kde_distribution(df, columns, filename="kde_distribution.jpg", out_dir="./evaluation/figures"):
    """
    绘制各变量的 KDE 分布
    每个子图一个colorbar
    """
    fig_dir = ensure_fig_dir(out_dir)

    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 5))
    if len(columns) == 1:
        axes = [axes]

    for i, col in enumerate(columns):
        ax = axes[i]
        if col not in df.columns:
            ax.text(0.5, 0.5, f"'{col}' not in df.columns", ha='center', va='center')
            continue

        # sns.kdeplot(df[col], ax=ax, fill=False, color="black")
        sns.kdeplot(df[col], ax=ax, fill=False, color="black", clip=(df[col].min(), df[col].max()))
        lines = ax.get_lines()
        if not lines:
            ax.set_title(f"No Data for {col}")
            continue

        line = lines[-1]
        x_plot, y_plot = line.get_xdata(), line.get_ydata()
        idxsort = np.argsort(x_plot)
        x_plot, y_plot = x_plot[idxsort], y_plot[idxsort]

        vmin = max(np.min(x_plot), df[col].min())   #发现负数
        vmax = min(np.max(x_plot), df[col].max())
        cmap = cm.get_cmap("coolwarm")
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        for j in range(len(x_plot)-1):
            x0, x1 = x_plot[j], x_plot[j+1]
            y0, y1 = y_plot[j], y_plot[j+1]
            color = cmap(norm((x0 + x1)*0.5))
            verts = np.array([
                [x0, 0],
                [x0, y0],
                [x1, y1],
                [x1, 0]
            ])
            poly = PolyCollection([verts], facecolors=[color], edgecolor='none', alpha=0.6)
            ax.add_collection(poly)

        ax.set_title(f"KDE of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.set_xlim(df[col].min(), df[col].max())

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax)
        cb.set_label("Value Range", fontweight='bold', fontsize=12)
        cb.ax.tick_params(labelsize=10)

    plt.tight_layout()
    save_path = os.path.join(fig_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[plot_kde_distribution] => {save_path}")


def plot_catalyst_size_vs_product(df, filename="catalyst_size_vs_product.jpg", out_dir="./evaluation/figures"):
    """
    绘制催化剂尺寸 vs 产物产量（散点图）
    x='Catalyst size', y=product, hue='Active metal'
    """
    fig_dir = ensure_fig_dir(out_dir)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    products = ['H2', 'CO', 'C1', 'C2+']

    for i, product in enumerate(products):
        ax = axes[i // 2, i % 2]
        needed_cols = ['Particle size (nm)', 'Active metal', product]
        if all(col in df.columns for col in needed_cols):
            sns.scatterplot(x='Particle size (nm)', y=product, hue='Active metal', data=df, ax=ax, alpha=0.7)
            ax.set_title(f'Particle size (nm) vs {product} Yield')
            ax.set_xlabel('Particle size (nm)')  # or mm if needed
            ax.set_ylabel(f'{product} Yield (%)')
        else:
            ax.text(0.5, 0.5, f"Cols not found for {product}", ha='center', va='center')
    plt.tight_layout()
    save_path = os.path.join(fig_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[plot_catalyst_size_vs_product] => {save_path}")


def plot_potential_vs_product_by_electrolyte(df, filename="potential_vs_product_by_electrolyte.jpg",
                                             out_dir="./evaluation/figures"):
    """
    绘制【电解质】分组下的电位 vs 产物散点, hue='Electrode support' 仅作为示例.
    """
    fig_dir = ensure_fig_dir(out_dir)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    products = ['H2', 'CO', 'C1', 'C2+']

    for i, product in enumerate(products):
        ax = axes[i // 2, i % 2]
        needed_cols = ['Potential (V vs. RHE)', 'Electrode support', product]
        if all(col in df.columns for col in needed_cols):
            sns.scatterplot(x='Potential (V vs. RHE)', y=product, hue='Electrode support',
                            data=df, ax=ax, alpha=0.7)
            ax.set_title(f'Potential vs {product} (Electrode support Hue)')
            ax.set_xlabel('Potential (V vs. RHE)')
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
    需求列: 'Active metal','Potential','H2','CO','C1','C2+'
    """
    fig_dir = ensure_fig_dir(out_dir)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    products = ['H2', 'CO', 'C1', 'C2+']

    if 'Potential (V vs. RHE)' in df.columns:
        df['Potential_bin'] = pd.cut(df['Potential (V vs. RHE)'], bins=5)
    else:
        df['Potential_bin'] = "Unknown"

    for i, product in enumerate(products):
        ax = axes[i]
        needed_cols = ['Active metal', product, 'Potential_bin']
        if all(col in df.columns for col in needed_cols):
            sns.boxplot(x='Active metal', y=product, hue='Potential_bin', data=df, ax=ax)
            ax.set_title(f'{product} by Active Metal & Potential')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5,0.5,f"Cols not found for {product}", ha='center', va='center')
    plt.tight_layout()
    save_path = os.path.join(fig_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[plot_product_distribution_by_catalyst_and_potential] => {save_path}")


# ============【新增】三张boxplot + 一张 potential vs product ============

def plot_product_vs_potential_bin(df, filename="box_product_vs_potential_bin.jpg", out_dir="./evaluation/figures"):
    """
    (1) 产物 vs Potential_bin 盒须图
    需求列: "Potential","H2","CO","C1","C2+"
    """
    fig_dir = ensure_fig_dir(out_dir)
    products = ["H2","CO","C1","C2+"]

    if 'Potential (V vs. RHE)' not in df.columns:
        print("[WARN] 'Potential (V vs. RHE)' not in df => skip boxplot")
        return

    # bin
    df['Potential_bin_custom'] = pd.cut(df['Potential (V vs. RHE)'], bins=5)
    fig, axes = plt.subplots(1, 4, figsize=(20,5))
    for i, product in enumerate(products):
        ax = axes[i]
        if product not in df.columns:
            ax.text(0.5,0.5,f"'{product}' not in df", ha='center', va='center')
            continue
        sns.boxplot(x='Potential_bin_custom', y=product, data=df, ax=ax)
        ax.set_title(f"{product} vs Potential Bin")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    out_path = os.path.join(fig_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot_product_vs_potential_bin] => {out_path}")


def plot_product_vs_shape(df, filename="box_product_vs_shape.jpg", out_dir="./evaluation/figures"):
    """
    (2) 产物 vs Shape 盒须图
    需求列: "Shape","H2","CO","C1","C2+"
    """
    fig_dir = ensure_fig_dir(out_dir)
    products = ["H2","CO","C1","C2+"]

    if 'Shape' not in df.columns:
        print("[WARN] 'Shape' not in df => skip boxplot shape")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20,5))
    for i, product in enumerate(products):
        ax = axes[i]
        if product not in df.columns:
            ax.text(0.5,0.5,f"'{product}' not in df", ha='center', va='center')
            continue
        sns.boxplot(x='Shape', y=product, data=df, ax=ax)
        ax.set_title(f"{product} vs Shape")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    out_path = os.path.join(fig_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot_product_vs_shape] => {out_path}")


def plot_product_vs_catalyst(df, filename="box_product_vs_catalyst.jpg", out_dir="./evaluation/figures"):
    """
    (3) 产物 vs Catalyst 盒须图
    需求列: "Catalyst","H2","CO","C1","C2+"
    """
    fig_dir = ensure_fig_dir(out_dir)
    products = ["H2","CO","C1","C2+"]

    if 'Active metal' not in df.columns:
        print("[WARN] 'Active metal' not in df => skip boxplot catalyst")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20,5))
    for i, product in enumerate(products):
        ax = axes[i]
        if product not in df.columns:
            ax.text(0.5,0.5,f"'{product}' not in df", ha='center', va='center')
            continue
        sns.boxplot(x='Active metal', y=product, data=df, ax=ax)
        ax.set_title(f"{product} vs Active metal")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    out_path = os.path.join(fig_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot_product_vs_catalyst] => {out_path}")


def plot_potential_vs_product(df, filename="three_dot_potential_vs_product.jpg", out_dir="./evaluation/figures"):
    """
    (4) 仅考虑 Potential vs. Product (H2/CO/C1/C2+) 三个点, 不区分别的hue
       这里随意三点 => 你可自定义
    """
    fig_dir = ensure_fig_dir(out_dir)
    products = ["H2","CO","C1","C2+"]
    if 'Potential (V vs. RHE)' not in df.columns:
        print("[WARN] 'Potential (V vs. RHE)' not in df => skip potential vs product")
        return

    plt.figure(figsize=(7,6))
    for product in products:
        if product in df.columns:
            plt.scatter(df['Potential (V vs. RHE)'], df[product], label=product, alpha=0.7)

    plt.title("Potential vs Products (3 dot style)")
    plt.xlabel("Potential (V vs. RHE)")
    plt.ylabel("Yield (%)")
    plt.legend()
    out_path = os.path.join(fig_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot_potential_vs_product] => {out_path}")


# ============ plot_advanced_data_analysis ============
def plot_advanced_data_analysis(df, out_dir):
    """
    调用若干函数做数据分析可视化(如KDE,散点,盒须图等).
    针对你的列: 'Potential','catalyst_size','electrolyte','H2','CO','C1','C2+','Catalyst','Shape', ...
    新增三张boxplot + 一个potential vs product散点
    """
    fig_dir = ensure_fig_dir(out_dir)

    # 1) KDE 分布
    possible_cols = [c for c in ["Potential (V vs. RHE)","H2","CO","C1","C2+","Particle size (nm)"] if c in df.columns]
    if len(possible_cols) > 0:
        plot_kde_distribution(df, possible_cols, filename="kde_distribution.jpg", out_dir=fig_dir)

    # 2) catalyst_size vs product
    plot_catalyst_size_vs_product(df, filename="catalyst_size_vs_product.jpg", out_dir=fig_dir)

    # 3) potential_vs_product_by_electrolyte
    plot_potential_vs_product_by_electrolyte(df, filename="potential_vs_product_by_electrolyte.jpg", out_dir=fig_dir)

    # 4) product_distribution_by_catalyst_and_potential
    plot_product_distribution_by_catalyst_and_potential(df, filename="product_distribution.jpg", out_dir=fig_dir)

    # ------------- 【新增】三张盒须图 + 一个潜势散点 -------------
    # (a) 产物 vs Potential_bin
    plot_product_vs_potential_bin(df, filename="box_product_vs_potential_bin.jpg", out_dir=fig_dir)
    # (b) 产物 vs Shape
    plot_product_vs_shape(df, filename="box_product_vs_shape.jpg", out_dir=fig_dir)
    # (c) 产物 vs Catalyst
    plot_product_vs_catalyst(df, filename="box_product_vs_catalyst.jpg", out_dir=fig_dir)
    # (d) potential vs product 三点图
    plot_potential_vs_product(df, filename="three_dot_potential_vs_product.jpg", out_dir=fig_dir)

    print("[INFO] plot_advanced_data_analysis => done.")
# --------------------------------------------------------

# ========== 推理可视化(2D Heatmap + Confusion)保留 ==========

def model_predict(model, X_2d):
    """
    根据传入的模型类型(是否是PyTorch)进行预测
    """
    if hasattr(model, 'eval') and hasattr(model, 'forward'):
        import torch
        model.eval()
        X_tensor = torch.tensor(X_2d, dtype=torch.float32)
        with torch.no_grad():
            out = model(X_tensor)
        return out.cpu().numpy()
    else:
        # sklearn / catboost / xgboost
        return model.predict(X_2d)


def plot_2d_mimo_heatmaps(
    grid_x, grid_y, predictions,
    out_labels=None,
    out_dir="./",
    prefix="mimo_heatmap"
):
    """
    多输出回归结果的 2D 网格热力图 (一次性画 out_dim 张图).
    predictions.shape = (H, W, out_dim)
    """
    H, W, out_dim = predictions.shape
    if out_labels is None or len(out_labels)!=out_dim:
        out_labels = [f"Output_{i+1}" for i in range(out_dim)]

    fig_dir = ensure_fig_dir(out_dir)

    for i in range(out_dim):
        plt.figure(figsize=(6,5))
        z = predictions[:,:,i]
        cm_ = plt.pcolormesh(grid_x, grid_y, z, shading='auto', cmap='viridis')
        cb = plt.colorbar(cm_)
        cb.set_label(f"{out_labels[i]}", fontsize=12)

        plt.xlabel("potential")
        plt.ylabel("catalyst_size")
        plt.title(f"Heatmap of {out_labels[i]}")
        out_fn = os.path.join(fig_dir, f"{prefix}_{i+1}.jpg")
        plt.savefig(out_fn, dpi=150, bbox_inches='tight')
        plt.close()


def plot_inference_2d_heatmap_for_two_continuous_vars(
        model, x_col_names, numeric_cols_idx, stats_dict,
        conti_var_x="potential", conti_var_y="catalyst_size",
        n_points=50,
        out_dir="./",
        y_col_names=None,
        scaler_x=None,
        scaler_y=None
):
    """
    (来自之前示例) ...
    """
    # ... 保留你之前的实现即可
    pass  # 省略示例


def plot_inference_confusion_matrix_for_two_categories(
        model, x_col_names, numeric_cols_idx, stats_dict,
        group1_index=0,
        group2_index=1,
        out_dir="./",
        y_col_names=None,
        scaler_x=None,
        scaler_y=None
):
    """
    (来自之前示例) ...
    """
    # ... 保留你之前的实现即可
    pass  # 省略示例
```

损失函数构建:

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

模型文件ann:

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

模型文件catboost

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

模型文件dt

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

模型文件rf

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

机器学习算法训练策略

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

深度学习算法训练策略

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
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

    no_improve_epochs = 0

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve_epochs = 0
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            no_improve_epochs += 1

        if early_stopping and no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best val_loss={best_val_loss:.6f} at epoch {best_epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model with val_loss={best_val_loss:.6f} from epoch {best_epoch}.")

    return model, train_losses, val_losses
```

main函数,用于读取需要训练的模型,以及相应的全部后处理逻辑,并出图:

```
"""
main.py

需求:
1) 同一个脚本支持:
   - K折 或 单次拆分
2) 模型创建统一在 create_model_by_type(...) 里
3) 训练/可视化
4) 保存模型 + scaler + colnames + 额外保存 metadata(包含 continuous_cols 的 min,max,mean 以及 onehot_groups)

【本版】:
- 保留你现有“K折 + 单次拆分 + 可视化 + 模型保存”全部逻辑。
- 支持 bounded_output => 0..100 -> -1..1 -> standard => ANN/RF/...
  并在预测后 inverse_transform_output => 回到 [0,100].
- 不再使用 logit transform。
- 不删减你的注释，只在必要处做增量修改。
"""

import yaml
import os
import numpy as np
import torch
import joblib

from data_preprocessing.data_loader import load_dataset, load_raw_data_for_correlation, extract_data_statistics
from data_preprocessing.data_split import split_data, kfold_split_data
from data_preprocessing.my_dataset import MyDataset
from data_preprocessing.scaler_utils import (
    standardize_data, inverse_transform_output, save_scaler
)
# 模型
from models.model_rf import RFRegression
from models.model_dt import DTRegression
from models.model_catboost import CatBoostRegression
from models.model_xgb import XGBRegression
from models.model_ann import ANNRegression
# 损失 & 训练脚本
from losses.torch_losses import get_torch_loss_fn
from trainers.train_torch import train_torch_model_dataloader
from trainers.train_sklearn import train_sklearn_model
# 指标 & 可视化
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
    # 新增
    plot_advanced_data_analysis
)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def create_model_by_type(model_type, config, random_seed=42, input_dim=None):
    """
    模型创建统一在这里:
      - ANN => 需input_dim
      - RF/DT/XGB/CatBoost => 不需
    """
    if model_type == "ANN":
        ann_cfg = config["model"]["ann_params"]
        actual_dim = input_dim if input_dim is not None else ann_cfg["input_dim"]
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
    data_corr_rel = os.path.join(csv_name, "DataCorrelation")
    model_comp_rel= os.path.join(csv_name, "model_comparison")

    ensure_dir(os.path.join("./evaluation/figures", data_corr_rel))
    ensure_dir(os.path.join("./evaluation/figures", model_comp_rel))

    # =========== 1) Load data ===========
    X, Y, numeric_cols_idx, x_col_names, y_col_names = load_dataset(csv_path)

    # =========== 1.5) 提取统计信息，写入 metadata.pkl ===========
    stats_dict = extract_data_statistics(X, x_col_names, numeric_cols_idx)

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

    # =========== 2.5) 数据分析可视化(可选) ===========
    if config["evaluation"].get("save_data_analysis_plots", False):
        if 'df_raw_14' in locals():
            out_dir = os.path.join("./evaluation/figures", data_corr_rel)
            plot_advanced_data_analysis(df_raw_14, out_dir=out_dir)
        else:
            print("[WARNING] df_raw_14 not loaded, skip advanced data plots.")

    use_k_fold    = config["data"].get("use_k_fold", False)
    k_folds       = config["data"].get("k_folds", 5)
    random_seed   = config["data"].get("random_seed", 42)
    model_types   = config["model"]["types"]
    save_overfit_bar = config["evaluation"].get("save_models_evaluation_bar", False)

    # 看是否要 bounded transform
    do_bounded = config["preprocessing"].get("bounded_output", False)
    print(f"[INFO] bounded_output={do_bounded}")

    # ------------------- K-Fold -------------------
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

                # 标准化 => 可能带 bounded => -1..1
                (X_tr_s, X_va_s, sx), (Y_tr_s, Y_va_s, sy) = standardize_data(
                    X_tr, X_va, Y_tr, Y_va,
                    do_input=config["preprocessing"]["standardize_input"],
                    do_output=config["preprocessing"]["standardize_output"],
                    numeric_cols_idx=numeric_cols_idx,
                    do_output_bounded=do_bounded
                )

                actual_input_dim= X_tr_s.shape[1]
                model= create_model_by_type(
                    model_type=mtype,
                    config=config,
                    random_seed=random_seed,
                    input_dim=actual_input_dim
                )

                if mtype in ["ANN"]:
                    loss_fn= get_torch_loss_fn(config["loss"]["type"])
                    train_ds= MyDataset(X_tr_s, Y_tr_s)
                    val_ds=   MyDataset(X_va_s, Y_va_s)

                    if mtype=="ANN":
                        ann_cfg= config["model"]["ann_params"]
                        lr_= float(ann_cfg["learning_rate"])
                        bs_= ann_cfg["batch_size"]
                        ep_= ann_cfg["epochs"]
                        wdecay= float(ann_cfg.get("weight_decay", 0.0))
                        ckp= None
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
                    model= train_sklearn_model(model, X_tr_s, Y_tr_s)
                    train_pred= model.predict(X_tr_s)
                    val_pred=   model.predict(X_va_s)

                # 反变换 => 回到 0..100
                if config["preprocessing"]["standardize_output"]:
                    train_pred= inverse_transform_output(train_pred, sy)
                    val_pred=   inverse_transform_output(val_pred, sy)

                # 计算原始尺度上的指标
                train_m= compute_regression_metrics(Y_tr, train_pred)
                val_m=   compute_regression_metrics(Y_va, val_pred)
                print(f"   Fold{fold_i} => train={train_m}, valid={val_m}")

                fold_train_list.append(train_m)
                fold_val_list.append(val_m)

                if config["evaluation"].get("save_scatter_mse_plot", False):
                    out_mse_tr= os.path.join(train_sub_rel, f"{mtype}_fold{fold_i}_mse_scatter_train.jpg")
                    plot_scatter_3d_outputs_mse(Y_tr, train_pred, None, filename=out_mse_tr)

                    out_mse_val= os.path.join(valid_sub_rel, f"{mtype}_fold{fold_i}_mse_scatter.jpg")
                    plot_scatter_3d_outputs_mse(Y_va, val_pred, None, filename=out_mse_val)

            train_mse= [fm["MSE"] for fm in fold_train_list]
            train_mae= [fm["MAE"] for fm in fold_train_list]
            train_r2 = [fm["R2"]  for fm in fold_train_list]
            avg_train_m= {
                "MSE": float(np.mean(train_mse)),
                "MAE": float(np.mean(train_mae)),
                "R2" : float(np.mean(train_r2))
            }

            val_mse= [fm["MSE"] for fm in fold_val_list]
            val_mae= [fm["MAE"] for fm in fold_val_list]
            val_r2 = [fm["R2"]  for fm in fold_val_list]
            avg_val_m= {
                "MSE": float(np.mean(val_mse)),
                "MAE": float(np.mean(val_mae)),
                "R2" : float(np.mean(val_r2))
            }
            print(f"   => K-Fold average: train={avg_train_m}, valid={avg_val_m}")
            kfold_train_metrics[mtype]= avg_train_m
            kfold_val_metrics[mtype]  = avg_val_m

        # 画 3 指标条形图
        if len(kfold_train_metrics) > 0:
            out_kf_tr = os.path.join(data_corr_rel, "three_metrics_horizontal_kfold_train.jpg")
            plot_three_metrics_horizontal(kfold_train_metrics, save_name=out_kf_tr)
        if len(kfold_val_metrics) > 0:
            out_kf_val= os.path.join(data_corr_rel, "three_metrics_horizontal_kfold_val.jpg")
            plot_three_metrics_horizontal(kfold_val_metrics, save_name=out_kf_val)

        if save_overfit_bar:
            if len(kfold_train_metrics)>0 and len(kfold_val_metrics)>0:
                overfit_kf = {}
                for m in kfold_train_metrics:
                    trm = kfold_train_metrics[m]
                    vam = kfold_val_metrics[m]
                    if trm["MSE"] == 0:
                        ms_ratio = float("inf")
                    else:
                        ms_ratio = vam["MSE"] / trm["MSE"]
                    r2_diff = trm["R2"] - vam["R2"]
                    overfit_kf[m] = {
                        "MSE_ratio": ms_ratio,
                        "R2_diff":   r2_diff
                    }
                out_kf_of = os.path.join(data_corr_rel, "overfitting_kfold.jpg")
                plot_overfitting_horizontal(overfit_kf, save_name=out_kf_of)

    # ------------------- 单次拆分 -------------------
    single_train_dict= {}
    single_val_dict= {}

    print("\n[INFO] Single-split => produce train/val images.")
    X_train, X_val, Y_train, Y_val= split_data(
        X, Y,
        test_size=config["data"]["test_size"],
        random_state=random_seed
    )

    (X_train_s, X_val_s, sx), (Y_train_s, Y_val_s, sy)= standardize_data(
        X_train, X_val, Y_train, Y_val,
        do_input=config["preprocessing"]["standardize_input"],
        do_output=config["preprocessing"]["standardize_output"],
        numeric_cols_idx=numeric_cols_idx,
        do_output_bounded=do_bounded
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
                checkpoint_path=ckp,
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
            model= train_sklearn_model(model, X_train_s, Y_train_s)
            train_pred= model.predict(X_train_s)
            val_pred=   model.predict(X_val_s)

        # 若输出也做标准化 => 反变换
        if config["preprocessing"]["standardize_output"]:
            train_pred= inverse_transform_output(train_pred, sy)
            val_pred=   inverse_transform_output(val_pred, sy)

        train_m= compute_regression_metrics(Y_train, train_pred)
        val_m  = compute_regression_metrics(Y_val,   val_pred)
        print(f"  [Train Metrics] => {train_m}")
        print(f"  [Valid Metrics] => {val_m}")

        single_train_dict[mtype]= train_m
        single_val_dict[mtype]  = val_m

        # 可视化
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

        if config["evaluation"].get("save_feature_importance_bar", False):
            if hasattr(model, "feature_importances_"):
                fi_out = os.path.join(model_sub_rel, f"{mtype}_feature_importance.jpg")
                plot_rf_feature_importance_bar(model, x_col_names, filename=fi_out)
            else:
                if hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
                    fi_out = os.path.join(model_sub_rel, f"{mtype}_feature_importance.jpg")
                    plot_rf_feature_importance_bar(model.model, x_col_names, filename=fi_out)

        # ========== 保存模型 ==========
        model_dir = os.path.join("./models", mtype)
        ensure_dir(model_dir)

        # 保存 scaler
        sx_path = os.path.join(model_dir, f"scaler_x_{mtype}.pkl")
        sy_path = os.path.join(model_dir, f"scaler_y_{mtype}.pkl")
        save_scaler(sx, sx_path)
        save_scaler(sy, sy_path)

        # 保存列名
        np.save(os.path.join(model_dir, "x_col_names.npy"), x_col_names)
        np.save(os.path.join(model_dir, "y_col_names.npy"), y_col_names)

        if mtype=="ANN":
            print(f"[INFO] ANN => checkpoint saved at: {ckp}")
        else:
            out_pkl = os.path.join(model_dir, "trained_model.pkl")
            joblib.dump(model, out_pkl)
            print(f"[INFO] Saved {mtype} => {out_pkl}")

    # => 画3指标条形图 (train, val)
    if len(single_train_dict) > 0:
        out_3_train= os.path.join(data_corr_rel, "three_metrics_horizontal_train.jpg")
        plot_three_metrics_horizontal(single_train_dict, save_name=out_3_train)

    if len(single_val_dict) > 0:
        out_3_val= os.path.join(data_corr_rel, "three_metrics_horizontal_val.jpg")
        plot_three_metrics_horizontal(single_val_dict, save_name=out_3_val)

    if save_overfit_bar:
        if len(single_train_dict) > 0 and len(single_val_dict) > 0:
            overfit_single= {}
            for m in single_train_dict:
                trm= single_train_dict[m]
                vam= single_val_dict[m]
                if trm["MSE"] == 0:
                    ms_ratio= float("inf")
                else:
                    ms_ratio= vam["MSE"] / trm["MSE"]
                r2_diff= trm["R2"] - vam["R2"]
                overfit_single[m]= {
                    "MSE_ratio": ms_ratio,
                    "R2_diff":   r2_diff
                }
            out_overfit_single= os.path.join(data_corr_rel, "overfitting_single.jpg")
            plot_overfitting_horizontal(overfit_single, save_name=out_overfit_single)

    # 保存 metadata.pkl (stats_dict) => 供推理阶段使用
    meta_path = os.path.join("./models", "metadata.pkl")
    joblib.dump(stats_dict, meta_path)
    print(f"[INFO] metadata saved => {meta_path}")

    print("\n[INFO] Single-split done.")
    print("Train metrics =>", single_train_dict)
    print("Val   metrics =>", single_val_dict)


if __name__ == "__main__":
    main()
```

inference,用于推断+绘图:

```
"""
inference.py

需求(最新,2023-xx-xx):
1) 对 two_continuous_vars 画 2D 热力图 => (a) plot_inference_2d_heatmap_for_two_continuous_vars
   - 其它连续量取均值
   - each onehot组合 => 预测(Scaled)->逆变换(Real)->累加->clip->存
2) 对 two_onehot_groups 画混淆矩阵 => (b) plot_inference_confusion_matrix_for_two_categories
   - 同样先在 scaled 预测-> 逆变换-> clip
   - 混淆矩阵**每格**大小=1/3(默认值),
     并让 colorbar / 字体不至于挤得太密

其余需求(进度条/tqdm, config.yaml, metadata.pkl 读取, etc.)维持.
"""

import os
import yaml
import numpy as np
import torch
import joblib
import warnings

from tqdm import trange
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PolyCollection
from data_preprocessing.scaler_utils import (
    load_scaler,
    inverse_transform_output
)
import math

# ================ 模型加载 & 通用预测 ================
from models.model_ann import ANNRegression
from models.model_rf import RFRegression
from models.model_dt import DTRegression
from models.model_catboost import CatBoostRegression
from models.model_xgb import XGBRegression


def load_inference_model(model_type, config):
    """
    在 ./models/<model_type>/ 下加载:
      x_col_names.npy, y_col_names.npy
      若 ANN => best_ann.pt, 否则 => trained_model.pkl
    """
    model_dir = os.path.join("./models", model_type)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"[ERROR] Directory not found => {model_dir}")

    x_col_path = os.path.join(model_dir, "x_col_names.npy")
    y_col_path = os.path.join(model_dir, "y_col_names.npy")
    if not (os.path.exists(x_col_path) and os.path.exists(y_col_path)):
        raise FileNotFoundError("[ERROR] x_col_names.npy or y_col_names.npy not found.")

    x_col_names = list(np.load(x_col_path, allow_pickle=True))
    y_col_names = list(np.load(y_col_path, allow_pickle=True))

    if model_type=="ANN":
        ann_cfg = config["model"]["ann_params"]
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
        # 尝试 weights_only
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(state_dict)
        net.eval()
        return net, x_col_names, y_col_names
    else:
        pkl_path = os.path.join(model_dir, "trained_model.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"[ERROR] {pkl_path} not found.")
        model = joblib.load(pkl_path)
        return model, x_col_names, y_col_names


def model_predict(model, X_2d):
    """
    通用预测接口: Torch / sklearn / catboost / xgb
    """
    if hasattr(model, 'eval') and hasattr(model, 'forward'):
        # Torch
        with torch.no_grad():
            t_ = torch.tensor(X_2d, dtype=torch.float32)
            out_ = model(t_)
        return out_.cpu().numpy()
    else:
        return model.predict(X_2d)

# ================ (a) 2D Heatmap => 2 continuous + combos, in real domain ================
def plot_inference_2d_heatmap_for_two_continuous_vars(
    model,
    x_col_names,
    numeric_cols_idx,
    stats_dict,
    conti_var_x="Potential (V vs. RHE)",
    conti_var_y="Particle size (nm)",
    n_points=20,
    out_dir="./",
    y_col_names=None,
    scaler_x=None,
    scaler_y=None
):
    os.makedirs(out_dir, exist_ok=True)

    cinfo_x = stats_dict["continuous_cols"][conti_var_x]
    cinfo_y = stats_dict["continuous_cols"][conti_var_y]

    x_vals = np.linspace(cinfo_x["min"], cinfo_x["max"], n_points)
    y_vals = np.linspace(cinfo_y["min"], cinfo_y["max"], n_points)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)

    base_vec = np.zeros(len(x_col_names), dtype=float)
    for cname, cstat in stats_dict["continuous_cols"].items():
        idx_ = x_col_names.index(cname)
        base_vec[idx_] = cstat["mean"]  # 其它连续 => mean

    # onehot groups => all combos
    oh_groups = stats_dict["onehot_groups"]
    group_choices = [grp for grp in oh_groups]
    all_combos = list(product(*group_choices))
    max_combo = 20000
    if len(all_combos)>max_combo:
        print(f"[WARN] combos => {len(all_combos)}, truncated => {max_combo}")
        all_combos = all_combos[:max_combo]

    # 先估 out_dim => 1 sample
    tmp_inp = base_vec.copy().reshape(1,-1)
    if scaler_x:
        tmp_inp[:, numeric_cols_idx] = scaler_x.transform(tmp_inp[:, numeric_cols_idx])
    tmp_out = model_predict(model, tmp_inp)
    if tmp_out.ndim==1:
        out_dim = len(tmp_out)
    else:
        out_dim = tmp_out.shape[1]

    H, W = grid_x.shape
    predictions = np.zeros((H,W,out_dim), dtype=float)

    row_iter = trange(H, desc="Heatmap Rows", ncols=100)
    for i in row_iter:
        for j in range(W):
            cur_vec = base_vec.copy()
            cur_vec[x_col_names.index(conti_var_x)] = grid_x[i,j]
            cur_vec[x_col_names.index(conti_var_y)] = grid_y[i,j]

            # sum in REAL domain
            sum_real = np.zeros(out_dim, dtype=float)
            for combo in all_combos:
                tmpv = cur_vec.copy()
                # zero out oh
                for grp_ in oh_groups:
                    for cid_ in grp_:
                        tmpv[cid_] = 0
                # activate
                for cid_ in combo:
                    tmpv[cid_] = 1

                # => scale X
                inp_2d = tmpv.reshape(1,-1)
                if scaler_x:
                    inp_2d[:, numeric_cols_idx] = scaler_x.transform(inp_2d[:, numeric_cols_idx])
                scaled_pred = model_predict(model, inp_2d)
                real_pred = inverse_transform_output(scaled_pred, scaler_y)
                sum_real += real_pred[0] if real_pred.ndim==2 else real_pred

            avg_real = sum_real / len(all_combos)
            # clip => [0,100]
            avg_real = np.clip(avg_real, 0, 100)
            predictions[i,j,:] = avg_real

    # =========== 绘图 =============
    for odx in range(out_dim):
        z_ = predictions[:,:,odx]
        plt.figure(figsize=(6,5))
        cm_ = plt.pcolormesh(grid_x, grid_y, z_, shading='auto', cmap='viridis')
        cb_ = plt.colorbar(cm_)
        if y_col_names and odx<len(y_col_names):
            cb_.set_label(y_col_names[odx], fontsize=9)
            plt.title(f"Heatmap of {y_col_names[odx]}")
        else:
            cb_.set_label(f"Output {odx}", fontsize=9)
            plt.title(f"Heatmap - Out {odx}")
        cb_.ax.tick_params(labelsize=8)

        plt.xlabel(conti_var_x, fontsize=9)
        plt.ylabel(conti_var_y, fontsize=9)

        outfn = os.path.join(out_dir, f"heatmap_output_{odx+1}.jpg")
        plt.savefig(outfn, dpi=150, bbox_inches='tight')
        plt.close()


# ================ (b) Confusion => each cell => sum in real domain => final clip ================
def plot_inference_confusion_matrix_for_two_categories(
    model,
    x_col_names,
    numeric_cols_idx,
    stats_dict,
    group1_index=0,
    group2_index=1,
    out_dir="./",
    y_col_names=None,
    scaler_x=None,
    scaler_y=None,
    cell_scale=1/3  # 新增: 每个cell边长=1/3 => 缩小
):
    """
    row= oh_groups[group1_index], col= oh_groups[group2_index].
    其余 group => combos => each => scale predict => inverse => sum => final avg clip

    cell_scale => e.g. 1/3 => grid spacing= 1/3 => cell更小
    """
    os.makedirs(out_dir, exist_ok=True)

    oh_groups = stats_dict["onehot_groups"]
    if group1_index>=len(oh_groups) or group2_index>=len(oh_groups):
        print(f"[ERROR] group1_index={group1_index} or group2_index={group2_index} out of range!")
        return

    grpA = oh_groups[group1_index]  # row
    grpB = oh_groups[group2_index]  # col
    rest_groups = []
    for i,g in enumerate(oh_groups):
        if i!=group1_index and i!=group2_index:
            rest_groups.append(g)

    row_cats = grpA
    col_cats = grpB

    base_vec = np.zeros(len(x_col_names), dtype=float)
    for cname, cstat in stats_dict["continuous_cols"].items():
        idx_ = x_col_names.index(cname)
        base_vec[idx_] = cstat["mean"]

    from itertools import product
    rest_combos = list(product(*rest_groups))
    max_combo=20000
    if len(rest_combos)>max_combo:
        print(f"[WARN] rest combos => {len(rest_combos)}, truncated => {max_combo}")
        rest_combos = rest_combos[:max_combo]

    # out_dim
    tmp_inp = base_vec.copy().reshape(1,-1)
    if scaler_x:
        tmp_inp[:, numeric_cols_idx] = scaler_x.transform(tmp_inp[:, numeric_cols_idx])
    tmp_out = model_predict(model, tmp_inp)
    if tmp_out.ndim==1:
        out_dim = len(tmp_out)
    else:
        out_dim = tmp_out.shape[1]

    n_rows = len(row_cats)
    n_cols = len(col_cats)
    mat_pred = np.zeros((n_rows,n_cols,out_dim), dtype=float)

    row_iter = trange(n_rows, desc="ConfMatrixRows", ncols=100)
    for i in row_iter:
        rcid = row_cats[i]
        for j in range(n_cols):
            ccid = col_cats[j]
            sum_real = np.zeros(out_dim, dtype=float)

            for combo in rest_combos:
                vec2 = base_vec.copy()
                # zero
                for grp_ in oh_groups:
                    for cid_ in grp_:
                        vec2[cid_] = 0
                # activate
                vec2[rcid] = 1
                vec2[ccid] = 1
                for colid_ in combo:
                    vec2[colid_] = 1

                inp_2d = vec2.reshape(1,-1)
                if scaler_x:
                    inp_2d[:, numeric_cols_idx] = scaler_x.transform(inp_2d[:, numeric_cols_idx])
                scaled_out = model_predict(model, inp_2d)
                real_out = inverse_transform_output(scaled_out, scaler_y)
                sum_real += real_out[0] if real_out.ndim==2 else real_out

            avg_real = sum_real / len(rest_combos)
            avg_real = np.clip(avg_real,0,100)
            mat_pred[i,j,:] = avg_real

    # =========== 画混淆矩阵(每cell => cell_scale) =============
    fig, ax = plt.subplots(figsize=(8,8))  # 可以固定大一点
    ax.set_title("Confusion-like MIMO (smaller cells)", fontsize=10)

    # 计算 min,max => 用于颜色
    allvals = mat_pred.reshape(-1, out_dim)
    out_mins = allvals.min(axis=0)
    out_maxs = allvals.max(axis=0)

    # 先画网格
    for r_ in range(n_rows+1):
        ax.axhline(r_*cell_scale, color='black', linewidth=1)
    for c_ in range(n_cols+1):
        ax.axvline(c_*cell_scale, color='black', linewidth=1)

    # 画 polygons
    cmaps= [plt.get_cmap("Reds"), plt.get_cmap("Blues"),
            plt.get_cmap("Greens"), plt.get_cmap("Oranges")]

    for i in range(n_rows):
        for j in range(n_cols):
            vals = mat_pred[i,j,:]
            # corners:
            # BL= (j*cell_scale, i*cell_scale)
            # BR= ((j+1)*cell_scale, i*cell_scale)
            # TL= (j*cell_scale, (i+1)*cell_scale)
            # TR= ((j+1)*cell_scale, (i+1)*cell_scale)
            BL= ( j*cell_scale, i*cell_scale )
            BR= ( (j+1)*cell_scale, i*cell_scale )
            TL= ( j*cell_scale, (i+1)*cell_scale )
            TR= ( (j+1)*cell_scale, (i+1)*cell_scale )

            Cx= j*cell_scale + cell_scale/2
            Cy= i*cell_scale + cell_scale/2
            center= (Cx,Cy+ cell_scale/2)  # 你如果想更靠上/居中都可

            # 这里按4块: (0)top-left, (1)top-right, (2)bottom-right, (3)bottom-left
            for odx in range(min(4,out_dim)):
                val_ = vals[odx]
                norm_ = matplotlib.colors.Normalize(
                    vmin=out_mins[odx],
                    vmax=out_maxs[odx]
                )
                color_ = cmaps[odx](norm_(val_))

                if odx==0:
                    poly = [TL, (Cx,Cy+cell_scale/2), TR]
                elif odx==1:
                    poly = [TR, (Cx,Cy+cell_scale/2), BR]
                elif odx==2:
                    poly = [BR, (Cx,Cy+cell_scale/2), BL]
                else:
                    poly = [BL, (Cx,Cy+cell_scale/2), TL]

                ax.add_patch(plt.Polygon(poly, facecolor=color_, alpha=0.9))

    # xlim,ylim
    ax.set_xlim(0, n_cols*cell_scale)
    ax.set_ylim(0, n_rows*cell_scale)
    ax.invert_yaxis()

    # ticks => cell center?
    # 先 row center => i+0.5 => i+0.5 => we do i+0.5 => i*cell_scale + cell_scale/2
    ax.set_xticks( [ (j+0.5)*cell_scale for j in range(n_cols)] )
    ax.set_yticks( [ (i+0.5)*cell_scale for i in range(n_rows)] )

    col_labels= [ x_col_names[cid] for cid in col_cats ]
    row_labels= [ x_col_names[cid] for cid in row_cats ]

    ax.set_xticklabels( col_labels, rotation=45, ha='right', fontsize=9 )
    ax.set_yticklabels( row_labels, fontsize=9 )

    # colorbars
    fig.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)
    for odx in range(min(4,out_dim)):
        # 每个 colorbar 占 0.05 宽
        cax_left= 0.72 + 0.06*odx
        cax = fig.add_axes([cax_left, 0.1, 0.03, 0.8])
        norm_ = matplotlib.colors.Normalize(vmin=out_mins[odx], vmax=out_maxs[odx])
        cb_ = matplotlib.colorbar.ColorbarBase(cax, cmap=cmaps[odx], norm=norm_)
        if y_col_names and odx<len(y_col_names):
            label_ = y_col_names[odx]
        else:
            label_ = f"Out {odx}"
        cb_.set_label(label_, fontsize=8)
        cb_.ax.tick_params(labelsize=8)

    outfn= os.path.join(out_dir, "confusion_matrix_mimo.jpg")
    plt.savefig(outfn, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion => {outfn}")


# ================ 主函数 => 读取 config + metadata => run ================
def main():
    with open("./configs/config.yaml","r") as f:
        config= yaml.safe_load(f)

    inf_models= config["inference"].get("models", [])
    if not inf_models:
        print("[INFO] No inference models => exit.")
        return

    meta_path= os.path.join("./models","metadata.pkl")
    if not os.path.exists(meta_path):
        print(f"[ERROR] metadata => {meta_path} missing.")
        return
    stats_dict= joblib.load(meta_path)

    for mtype in inf_models:
        print(f"\n========== Inference for {mtype} ==========")
        try:
            model, x_col_names, y_col_names = load_inference_model(mtype, config)
        except FileNotFoundError as e:
            print(e)
            continue

        # load scaler
        model_dir= os.path.join("./models", mtype)
        sx_path= os.path.join(model_dir, f"scaler_x_{mtype}.pkl")
        sy_path= os.path.join(model_dir, f"scaler_y_{mtype}.pkl")
        if os.path.exists(sx_path):
            scaler_x= load_scaler(sx_path)
        else:
            scaler_x= None
        if os.path.exists(sy_path):
            scaler_y= load_scaler(sy_path)
        else:
            scaler_y= None

        # 1) heatmap
        out_dir1= os.path.join(f"{mtype}_inference","2d_heatmap")
        os.makedirs(out_dir1, exist_ok=True)
        conti_x= config.get("inference",{}).get("conti_x","Potential (V vs. RHE)")
        conti_y= config.get("inference",{}).get("conti_y","Particle size (nm)")

        numeric_cols_idx=[]
        for c_ in stats_dict["continuous_cols"].keys():
            numeric_cols_idx.append( x_col_names.index(c_) )

        plot_inference_2d_heatmap_for_two_continuous_vars(
            model=model,
            x_col_names=x_col_names,
            numeric_cols_idx=numeric_cols_idx,
            stats_dict=stats_dict,
            conti_var_x=conti_x,
            conti_var_y=conti_y,
            n_points=20,
            out_dir=out_dir1,
            y_col_names=y_col_names,
            scaler_x=scaler_x,
            scaler_y=scaler_y
        )

        # 2) confusion
        out_dir2= os.path.join(f"{mtype}_inference","confusion_matrix")
        os.makedirs(out_dir2, exist_ok=True)

        if len(stats_dict["onehot_groups"])>=2:
            plot_inference_confusion_matrix_for_two_categories(
                model,
                x_col_names,
                numeric_cols_idx,
                stats_dict,
                group1_index=0,
                group2_index=1,
                out_dir=out_dir2,
                y_col_names=y_col_names,
                scaler_x=scaler_x,
                scaler_y=scaler_y,
                cell_scale=1/3  # 每格 =1/3
            )
        else:
            print("[WARN] not enough oh groups => skip confusion.")


if __name__=="__main__":
    main()
```

我现在有几个要求:



1.我的所有绘图依赖的基本代码在我的visualization.py里面,但是一般来说这些代码应该在我的utils.py中.希望你帮我做好修改.

2.我的main函数中现在包括了训练和后处理两个逻辑,但是一般代码框架中应该是train.py做出所有的训练,以及相关重要数据的保存,然后用visualization,来读取数据并绘图.请您把相关的数据保存到一个特定的文件夹中,然后再运行visualization.py来读取config文件中所需要的后处理(包含train和inference)然后绘图.没错,也就是说我现在的inference文件中的推断,应当把数据保存好,然后visualization中可以读取并绘制,请把train和inference两个数据保存到postprocessing/数据集/train/model和postprocessing/数据集/inference/model中这些东西都按照需求改变,然后原来我的推理可视化结果保存在了

model_inference里面,现在保存在我的evaluation/figures/数据集/inference/model里面.由此一来,我的可视化和训练,推断就可以分开了,要不然每次修改图像的操作都要运行一个多小时,对我的代码修改带来了很大的问题.

3.我的inference中的绘图逻辑中(在你的修改后的代码应该在你的新的utils里面修改,然后在新的visualization里面调用),这里面的推断colorbar, 应该分别针对于四个预测数据的最小,最大值.就像其余连续性数据那样读取原始数据的最小最大值, 可以相应地给一个max(0, min_lim\*0.75), min(100, max_lim\*1.25)这样来进行控制colorbar的可视化.这里可能涉及到一些保存,请做好保存.

4.请不要过分删除我代码中的注释,尽量保持代码的原始逻辑,可以做适当简化,请给我所有的完整的代码.要完整的,且可以一次运行成功.我里面可视化的逻辑部分除了我让你修改的以外务必保持,包括命名,格式等.
