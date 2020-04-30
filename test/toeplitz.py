import numpy as np
from scipy.signal import convolve2d, correlate2d
from scipy.linalg import circulant,toeplitz

image = np.ones((25,1)).reshape(5,5)
print(image)

pad_image = np.append([[0]*5], image, axis=0)
pad_image = np.append(pad_image, [[0]*5], axis=0)
pad_image = np.append(pad_image, [[0]]*7, axis=1)
pad_image = np.append([[0]]*7, pad_image, axis=1)


def makeConvolver(image, kernel):
    permuter = np.asarray([i for i in range(0,image.size)]).reshape(image.shape)
    directions = [np.roll(np.roll(permuter, -1, axis=1), -1, axis=0), #sw
    np.roll(permuter, -1, axis=1), #w
    np.roll(np.roll(permuter, -1, axis=1), 1, axis=0), #nw
    np.roll(permuter, -1, axis=0), #s
    permuter, #c
    np.roll(permuter, 1, axis=0), #n
    np.roll(np.roll(permuter, 1, axis=1), -1, axis=0), #se
    np.roll(permuter, 1, axis=1), #e
    np.roll(np.roll(permuter, 1, axis=1), 1, axis=0)] #ne
    return sum([np.identity(image.size)[directions[i]] * kernel[i] for i in range(0,9)]).reshape(image.size, image.size)

A = makeConvolver(np.ones((3,3)), np.asarray([0,1,2,3,4,5,6,7,8]))
print(A)
