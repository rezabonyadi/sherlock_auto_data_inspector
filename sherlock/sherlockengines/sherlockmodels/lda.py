import numpy as np
import scipy as sp

class LDA():
    """The Linear Discriminant Analysis classifier, also known as Fisher's linear discriminant.
    Can besides from classification also be used to reduce the dimensionaly of the dataset.
    """
    def __init__(self):
        self.w = None
        self.n_components = 2

    def transform(self, X, y):
        self.fit(X, y)
        # Project data onto vector
        X_transform = X.dot(self.w)
        return X_transform

    def fit(self, X, y):
        # Separate data by class
        c_unique = np.unique(y)
        means = np.zeros((X.shape[1], X.shape[1]))
        mean_tot = X.mean(0)
        covs = np.zeros((X.shape[1], X.shape[1]))
        for c in c_unique:
            c_x = X[y == c]
            covs += np.cov(c_x.T)
            c_mean = c_x.mean(0)
            temp = (c_mean - mean_tot).reshape(-1, 1)
            means += (c_x.shape[0] * temp.dot(temp.T))

        eigen_values, eigen_vectors = sp.linalg.eig(covs, means)

        # Determine the vector which when X is projected onto it best separates the
        # data by class. w = (mean1 - mean2) / (cov1 + cov2)
        self.w = eigen_vectors[:,:self.n_components]

    def predict(self, X):
        y_pred = []
        for sample in X:
            h = sample.dot(self.w)
            y = 1 * (h < 0)
            y_pred.append(y)
        return y_pred

    def calculate_covariance_matrix(self, X, Y=None):
        """ Calculate the covariance matrix for the dataset X """
        if Y is None:
            Y = X
        n_samples = np.shape(X)[0]
        covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

        return np.array(covariance_matrix, dtype=float)
