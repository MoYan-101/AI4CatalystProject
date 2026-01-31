def train_sklearn_model(model,
                        X_train, Y_train,
                        X_val=None, Y_val=None,
                        enable_early_stop: bool = False,
                        es_rounds: int = 50):
    """
    Train a sklearn‑style model (RF / DT) 或自定义薄包装 (CatBoostRegression / XGBRegression).
    """

    if enable_early_stop and X_val is not None:
        fit_kwargs = dict(
            eval_set=[(X_val, Y_val)],
            early_stopping_rounds=es_rounds,
            verbose=False
        )
        # CatBoost 需要 use_best_model
        if hasattr(model, "model") and model.model.__class__.__name__.startswith("CatBoost"):
            fit_kwargs["use_best_model"] = True

        # 对 XGBRegression 等，也能正常 set_params 中的 early_stopping_rounds
        model.fit(X_train, Y_train, **fit_kwargs)
    else:
        model.fit(X_train, Y_train)

    return model
