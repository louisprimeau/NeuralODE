from mult import mult
import numpy as np

def transform(X):
    N,M = X.shape
    W = np.exp(-2j * np.pi * np.arange(N).reshape((N,1)) * np.arange(N))
    return(np.dot(np.dot(W,X),W))
