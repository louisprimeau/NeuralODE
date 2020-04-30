import numpy as np
from scipy.signal import convolve2d

def theta_slice(theta, part="kernel"):
    if(part=="kernel"):
        return(theta[0:9,:])
    else:
        return(theta[9:,:])

def convolve(t, image, theta):
    kernel = theta_slice(theta,part="kernel")
    image = image.reshape(int(np.sqrt(image.size)), int(np.sqrt(image.size)))
    kernel = kernel.reshape(int(np.sqrt(kernel.size)), int(np.sqrt(kernel.size))) # RESHAPE
    output = convolve2d(image, kernel, mode="same", boundary="wrap", fillvalue=0) # CONVOLVE
    output = output.reshape(output.size, 1) # RESHAPE
    return output

def F(t, image, theta):
    return relu(convolve(t,image,theta))

def euler_forward(f, t0, t1, x0, N, theta):
    h = (t1 - t0) / N
    x = [x0]
    for i in range(N):
        x.append(x[i] + h * f(t0 + h*i, x[i], theta))
    [xi.reshape(xi.size,1) for xi in x]
    return(x)

def euler_backward(f, t0, t1, x0, N, theta):
    h = (t1 - t0) / N
    x = [x0]
    kernel = theta_slice(theta,part="kernel")
    for i in range(N):
        A = d_z_convolve(x[i], kernel)
        x.append(np.dot(np.linalg.inv(np.identity(A.shape) - h*A), x[i]))
    [xi.reshape(xi.size,1) for xi in x]
    return(x)

def relu(x):
    return np.maximum(x, 0, x)

def linear(x, theta):
    m = theta_slice(theta, part="matrix")
    m = m.reshape(x.size, -1) # RESHAPE
    x = x.reshape(1, x.size)
    output = np.dot(x, m) # PREDICT
    output = output.reshape(output.size, 1) # RESHAPE
    return output

def imax(x):
    return np.argmax(x)

def L2(x, x_):
    return np.sum(np.square(x - x_.reshape(x.shape)))
