import math
import numpy as np

def get_sigmoid(x: float):
	return 1 / (1+np.exp(-x))

def calculate_mse(labels: list[int], predictions: list[int]):
	errors = []
	for i,j in zip(labels,predictions):
		errors.append((i-j)**2)
	return sum(errors)/len(errors)

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	# multiply features by current weights
	weighted_features = np.dot(features, weights) + bias

	probabilities = []
	for i in weighted_features:
		probabilities.append(get_sigmoid(i))

	mse = calculate_mse(labels, probabilities)

	return probabilities, mse
