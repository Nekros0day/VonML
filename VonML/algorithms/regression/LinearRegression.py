import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, lambda_val=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_val = lambda_val
        self.weights, self.bias = None, None
        self.history = {'loss': []}

    def regularize(self, weights):
        if self.regularization == "l1":
            return self.lambda_val * np.sign(weights)
        elif self.regularization == "l2":
            return self.lambda_val * weights
        elif self.regularization == "elasticnet":
            return self.lambda_val * (0.5 * weights + 0.5 * np.sign(weights))
        else:
            return 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            loss = self.mean_squared_error(y, y_predicted) + self.lambda_val * np.sum(self.weights**2)
            self.history['loss'].append(loss)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + self.regularize(self.weights)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
