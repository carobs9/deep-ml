import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	eigenvalues, eigenvectors = np.linalg.eig(matrix)
	return eigenvalues.real

