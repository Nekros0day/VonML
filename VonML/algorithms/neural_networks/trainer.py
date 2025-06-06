class SGDTrainer:
    """Stochastic gradient descent trainer."""

    def __init__(self, model, loss, loss_derivative, learning_rate=0.01):
        self.model = model
        self.loss = loss
        self.loss_derivative = loss_derivative
        self.learning_rate = learning_rate
        self.history = []

    def train(self, X, y, epochs=1000):
        for _ in range(epochs):
            preds = self.model.forward(X)
            loss_val = self.loss(y, preds)
            grad = self.loss_derivative(y, preds)
            self.model.backward(grad, self.learning_rate)
            self.history.append(loss_val)
        return self.history

