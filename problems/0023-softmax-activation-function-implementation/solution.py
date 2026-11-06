import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    exponentials = []
    outputs = []
    for i in scores:
        i = i - max(scores) # allowing for numerical stability
        exponentials.append(np.exp(i))
    sum_of_exponentials = sum(exponentials)
    for i in exponentials:
        outputs.append((i / sum_of_exponentials)+1e-5)
    return outputs