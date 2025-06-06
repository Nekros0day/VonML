import matplotlib.pyplot as plt
from VonML.algorithms.regression import LinearRegression
from VonML.datasets import make_regression_dataset


def main():
    X, y = make_regression_dataset(n_samples=100, noise=0.2, random_state=0)
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    preds = model.predict(X)
    plt.scatter(X[:, 0], y, label="data")
    plt.plot(X[:, 0], preds, color="red", label="prediction")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

