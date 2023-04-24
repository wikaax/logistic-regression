import numpy as np

# initialize sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, lr=0.01, n_iters=1000):     # learningRate (0.1 -> 0.0001), number of iterations
        self.lr = lr;
        self.n_iters = n_iters;
        self.weights = None;
        self.bias = None;

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)     # Initialize weights as zero
        self.bias = 0                           # Initialize bias as zero

        # Prediction
        for _ in range(self.n_iters):
            linear_predictions = np.dot(X, self.weights) + self.bias    # linear regression
            predictions = sigmoid(linear_predictions)                   # linear regression in sigmoid function

            # Calculating the gradient
            # np.dot already does a sum and result is one number
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))         # gradient for weights
            db = (1/n_samples) * np.sum(predictions - y)                # gradient for bias

            # Updating weights and bias
            self.weights = self.weights - self.lr * dw
            self.bias =  self.bias - self.lr * db

    def predict(self, X):
        linear_predictions = np.dot(X, self.weights) + self.bias
        y_predictions = sigmoid(linear_predictions)
        class_predictions = [0 if y<=0.5 else 1 for y in y_predictions]
        return class_predictions

    def get_params(self, deep=True):
        return {'lr': self.lr, 'n_iters': self.n_iters}