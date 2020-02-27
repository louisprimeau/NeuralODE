import numpy as np

def mult(A, B):
    product = np.dot(A, B)
    product = product * np.random.normal(loc=1.0, scale = 0.05, size=1)
    return product
