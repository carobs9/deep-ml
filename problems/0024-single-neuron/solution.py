import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # turn data into PyTorch stuff
    features = torch.FloatTensor(features)
    weights = torch.FloatTensor(weights)
    labels = torch.FloatTensor(labels)
    # - torch.matmul() for linear combination
    weighted_features = torch.matmul(features, weights) + bias
    # - torch.sigmoid() for activation
    probabilities = torch.sigmoid(weighted_features)
    # - torch.nn.functional.mse_loss() for MSE
    loss = torch.nn.functional.mse_loss(labels, probabilities)

    probabilities = [tensor.item() for tensor in probabilities]
    loss = loss.item()
    return probabilities, loss