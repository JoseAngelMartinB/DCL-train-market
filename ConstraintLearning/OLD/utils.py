import numpy as np

def smape(y_true, y_pred, epsilon=1e-8):
    """
    Computes the Symmetric Mean Absolute Percentage Error (sMAPE)
    between true and predicted values.

    Parameters:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values
    - epsilon: small value to avoid division by zero

    Returns:
    - smape: float, symmetric mean absolute percentage error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.abs(y_pred - y_true) / (denominator + epsilon)) * 100
    return smape
