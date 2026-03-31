import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _loss(y, y_hat):
    """Compute the Binary cross entropy loss.
    """
    n = len(y)
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / n

def _gradients(X, y, y_hat):
    """Compute the gradients of the loss w.r.t. w and b."""
    n = len(y)
    dw = np.dot(X.T, (y_hat - y)) / n
    db = np.sum(y_hat - y) / n
    return dw, db

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    w = np.zeros(X.shape[1])
    b = 0
    
    for step in range(steps):
        y_hat = _sigmoid(np.dot(X, w) + b)
        dw, db = _gradients(X, y, y_hat)
        w -= lr * dw
        b -= lr * db
    return w, b