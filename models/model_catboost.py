from catboost import CatBoostRegressor

class CatBoostRegression:
    """
    Wrapper for CatBoostRegressor to align with the scikit-learn style usage.
    """
    def __init__(self, iterations=100, learning_rate=0.1, depth=6, random_seed=42):
        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_seed,
            verbose=0,
            loss_function="MultiRMSE"  # <--- 关键：指定多输出回归损失
        )

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.get_feature_importance()
