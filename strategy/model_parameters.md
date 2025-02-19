 

```yaml
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
```
