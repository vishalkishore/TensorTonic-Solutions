import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    for _ in range(steps):

        # Forward pass
        z = X @ w + b
        y_hat = _sigmoid(z)

        # Prediction error
        error = y_hat - y

        # Gradients
        dw = (X.T @ error) / n_samples
        db = np.mean(error)

        # Gradient descent
        w -= lr * dw
        b -= lr * db

    return w, b