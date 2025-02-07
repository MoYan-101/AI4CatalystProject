"""
models/model_dt.py

A simple Decision Tree regressor wrapper using scikit-learn.
"""

from sklearn.tree import DecisionTreeRegressor

class DTRegression:
    """
    A simple regression model using DecisionTree from scikit-learn.
    """
    def __init__(self, max_depth=None, random_state=42):
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_
