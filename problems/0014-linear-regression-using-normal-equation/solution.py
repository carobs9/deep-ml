import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X = np.array(X)
    y = np.array(y)
    
    X_transposed = X.T
    first_term = np.dot(X_transposed, X)
    inverse = np.linalg.inv(first_term)
    
    # Correct formula: (X^T X)^(-1) * X^T * y
    theta = inverse @ X_transposed @ y
    
    return [np.round(x) for x in theta]