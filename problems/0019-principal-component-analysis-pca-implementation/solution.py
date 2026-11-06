import numpy as np
from numpy import linalg as LA

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # standardize
    standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0) # (3,2)
    # cov matrix
    cov_matrix = np.cov(np.transpose(standardized_data)) # (2,2)
    # eigen decomposition
    eva, eve = LA.eigh(cov_matrix)
    # sort eigenvalues and eigenvectors (columns!) descending
    eva_descending_indices = eva.argsort()[::-1]
    eva_elements_descending = eva[eva_descending_indices]
    eve_elements_descending = eve[:, eva_descending_indices]
    # keep the top k eigenvectors
    eve_top_k = eve_elements_descending[:, :k]

    for j in range(k):
        v = eve_top_k[:, j]
        idx = np.argmax(np.abs(v) > 1e-10)
        if v[idx] < 0:
            eve_top_k[:, j] = -v
            
    # project the data (not needed for this problem)
    principal_component = np.dot(standardized_data, eve_top_k)
    
    return eve_top_k

    
