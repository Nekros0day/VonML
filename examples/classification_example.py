import matplotlib.pyplot as plt
from VonML.algorithms.neural_networks import MLP, SGDTrainer, LOSSES
from VonML.datasets import make_classification_dataset


def main():
    X, y = make_classification_dataset(n_samples=200, random_state=0)
    mlp = MLP(layer_sizes=[2, 5, 1], activations=["tanh", "sigmoid"], rng=0)
    loss, loss_deriv = LOSSES["cross_entropy"]
    trainer = SGDTrainer(mlp, loss, loss_deriv, learning_rate=0.1)
    trainer.train(X, y, epochs=1000)
    preds = mlp.forward(X) > 0.5
    plt.scatter(X[:, 0], X[:, 1], c=preds[:, 0], cmap="bwr", marker="o")
    plt.title("Classification result")
    plt.show()


if __name__ == "__main__":
    main()

