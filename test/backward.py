import numpy as np
from forward import convolve

def theta_slice(theta, part="kernel"):
    if(part=="kernel"):
        return(theta[0:9,:])
    else:
        return(theta[9:,:])

def d_t_convolve(image):
    permuter = np.asarray([i for i in range(0,image.size)]).reshape(image.shape)
    directions = [np.roll(permuter, (-1,-1), axis=(0,1)), #sw
                  np.roll(permuter, -1, axis=1), #w
                  np.roll(permuter, (1,-1), axis=(0,1)), #nw
                  np.roll(permuter, -1, axis=0), #s
                  permuter, #c
                  np.roll(permuter, 1, axis=0), #n
                  np.roll(permuter, (-1,1), axis=(0,1)), #se
                  np.roll(permuter, 1, axis=1), #e
                  np.roll(permuter, (1,1), axis=(0,1))] #n
    return np.concatenate([np.dot(np.identity(image.size)[directions[i]], image) for i in range(0,len(directions))], axis=1).reshape(image.size,9)

def d_z_convolve(image, kernel): #derivative of 3x3 convolution wrt to input
    size = (int(image.size**0.5),)*2
    permuter = np.asarray([i for i in range(0,image.size)]).reshape(size)
    directions = [np.roll(permuter, (-1,-1), axis=(0,1)), #sw
                  np.roll(permuter, (0,-1) , axis=(0,1)), #w
                  np.roll(permuter, (1,-1) , axis=(0,1)), #nw
                  np.roll(permuter, (-1,0) , axis=(0,1)), #s
                  permuter, #c
                  np.roll(permuter, (1,0)  , axis=(0,1)), #n
                  np.roll(permuter, (-1,1) , axis=(0,1)), #se
                  np.roll(permuter, (0,1)  , axis=(0,1)), #e
                  np.roll(permuter, (1,1)  , axis=(0,1))] #ne
    return sum([np.identity(image.size)[directions[i].reshape(image.size,1)] * kernel[i] for i in range(0,9)]).reshape(image.size, image.size)

def dFdt(image):
    return(np.dot(d_relu(image), d_t_convolve(image)))

def dFdz(t, image, theta):
    kernel = theta_slice(theta, part="kernel")
    return(np.dot(d_relu(convolve(t, image, theta)), d_z_convolve(image,kernel)))

def d_relu(input):
    x = np.heaviside(input, 0)
    x = np.diag(x.flatten())
    return x

def d_linear(x, theta):
    return theta_slice(theta, part="matrix").reshape(-1, x.size)

def d_L2(input, label):
    return np.transpose(2 * (input - label))
