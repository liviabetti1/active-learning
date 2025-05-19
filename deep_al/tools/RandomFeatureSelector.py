from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class RandomFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, num_features, random_state=None):
        self.num_features = num_features
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        print(f"Random State: {self.random_state}")
        n_features = X.shape[1]
        self.selected_indices_ = rng.choice(n_features, self.num_features, replace=False)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]