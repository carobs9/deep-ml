import numpy as np

def sigmoid_activation(x: int):
	return 1/ (1 + np.exp(-x))

def get_mse(labels, predictions):
	errors = []
	for i,j in zip(labels, predictions):
		errors.append((i-j)**2)

	return sum(errors) / len(labels)	

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	weights = initial_weights
	bias = initial_bias
	mse_values = []

	m,n = features.shape

	for _ in range(epochs):
		# 1. forward pass (z = X·w + b) and predictions (sigmoid(z))
		weighted_features = np.dot(features, weights) + bias
		predictions = sigmoid_activation(weighted_features)
		# 2. compute MSE and append to mse_values
		mse_values.append(get_mse(labels, predictions))
		# 3. compute gradients for weights and bias
		delta = (2/m)*(predictions-labels)*predictions*(1-predictions)
		gradient = np.dot(np.transpose(features), delta)
		gradient_bias = np.sum(delta)
		# 4. update weights and bias
		weights = weights - learning_rate * gradient
		bias = bias - learning_rate * gradient_bias

	return weights, bias, mse_values