import numpy as np

# Initialize sigmoid function
# s(x) = 1/(1+e^(-x))
def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    # learningRate (0.1 -> 0.0001, numberOfIterations
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr;
        self.n_iters = n_iters;
        self.weights = None;
        self.bias = None;

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weight as zero
        self.bias = 0;                          # Initialize bias as zero

        for _ in range(self.n_iters):
            linear_predictions = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_predictions)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.lr*dw
            self.bias =  self.bias - self.lr*db

    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        y_predictions = sigmoid(linear_predictions)
        class_pred = [0 if y<=0.5 else 1 for y in y_predictions]   # for every y in y_pred
        return class_pred