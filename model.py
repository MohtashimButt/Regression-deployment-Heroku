import numpy as np

class CustomLinearRegression:
    def __init__(self, learning_rate=0.1, num_epochs=50):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None
        self.train_losses = []
        self.test_losses = []

    def fit(self, X_train, y_train, X_test, y_test):
        np.random.seed(0)
        self.weights = np.random.rand(X_train.shape[1])
        self.bias = np.random.rand()

        for epoch in range(self.num_epochs):
            # Forward pass
            y_pred_train = np.dot(X_train, self.weights) + self.bias
            y_pred_test = np.dot(X_test, self.weights) + self.bias

            # Calculate MSE loss
            train_loss = np.mean((y_pred_train - y_train) ** 2)
            test_loss = np.mean((y_pred_test - y_test) ** 2)

            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)

            # Gradient descent update
            gradient = np.dot(X_train.T, (y_pred_train - y_train)) / X_train.shape[0]
            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * np.mean(y_pred_train - y_train)

            print(f"Epoch {epoch+1}/{self.num_epochs}, Training Loss: {train_loss}, Testing Loss: {test_loss}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
