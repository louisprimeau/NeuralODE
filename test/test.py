
from scipy.signal import convolve2d
import numpy as np
import random
np.set_printoptions(linewidth=100)

"""
PARAMETERS (THETA)
"""

np.random.seed(0)
kernel = np.random.normal(scale=(2/784)**0.5,size=(3,3))
eps = np.finfo(float).eps ** 3
kernel = kernel.astype("complex")
kernel[0,0] = complex(kernel[0,0],eps)

matrix = np.random.rand(784,10) / 1000
c_theta = np.concatenate((kernel.reshape(kernel.size,1), matrix.reshape(matrix.size,1)))
r_theta = c_theta.real

t0 = 0
t1 = 2
N = t1 - t0

lr = 0.0001
num_epochs = 1

"""
DATA
"""

from data import data
images, labels = data("../neuralODE/RKNet/data/train/")
images = images[0:100]
labels = labels[0:100]

"""
FORWARD
"""

from forward import *

"""
BACKWARD
"""

from backward import *

"""
ADJOINT
"""

def adj_euler_forward(f, t0, t1, x0, N, theta, ss): #identical to the other one, but takes in ss for correct adjoint slicing
    h = (t1 - t0) / N
    x = [x0]
    for i in range(N):
        x.append(x[i] + h * f(t0 + h*i, x[i], theta, ss))
    [xi.reshape(xi.size,1) for xi in x]
    return(x)

def adjoint(f, theta, t0, t1, zt1, dLdz1, x_t):
    s0 = np.concatenate((dLdz1, np.zeros((9, 1))))
    ss = [dLdz1.size, dLdz1.size + 9]
    def aug_dynamics(t, s, theta, ss):
        r1 = -np.dot(np.transpose(s[0:ss[0],:]), dFdz(t, x_t[int(t)], theta))
        r2 = -np.dot(np.transpose(s[0:ss[0],:]), dFdt(x_t[int(t) - 1]))
        return(np.transpose(np.concatenate((r1,r2), axis=1)))
    s1 = adj_euler_forward(aug_dynamics, t1, t0, s0, N, theta, ss)
    return s1[len(s1) - 1][ss[0]:ss[1]]

def dLdz(z, linear_output, relu_output, labels, theta):
    return(np.dot(d_L2(linear_output, labels.reshape(-1,1)), np.dot(d_linear(relu_output, theta), d_relu(z))).reshape(-1,1))


"""
Training
"""
epoch_losses = []
for i in range(0,num_epochs):
    epoch_loss = 0
    print("---------------")
    print("Epoch", i)
    order = list(zip(images,labels))
    random.shuffle(order)
    counter = 1
    mean_derivative = None
    for image, label in order:

        """
        FORWARD PREDICTION
        """
        theta = c_theta
        x_t = euler_forward(convolve, t0, t1, image, N, theta)
        r_t = relu(x_t[len(x_t) - 1])
        l_t = linear(r_t, theta)
        loss = L2(l_t, label)

        real_derivative =  loss.imag / eps

        theta = r_theta
        x_t = euler_forward(convolve, t0, t1, image, N, theta)
        r_t = relu(x_t[len(x_t) - 1])
        l_t = linear(r_t, theta)
        loss = L2(l_t, label)
        epoch_loss += loss

        """
        BACKWARD PROPAGATION
        """

        # Make derivatives:
        # Function is L2(linear(relu(euler_forward(convolve))))
        z1 = x_t[len(x_t) - 1].reshape(-1,1)
        dLdz1 = dLdz(z1, l_t, r_t, label, theta)
        theta_d = adjoint(convolve, theta, t0, t1, z1, dLdz1, x_t)
        lineard = d_L2(l_t, label)
        lineard = np.concatenate([lineard[0,i]*np.ones((image.size,1)) for i in range(label.size)])
        derivative = np.concatenate((theta_d.reshape(-1,1), lineard.reshape(-1,1)))

        print(real_derivative, derivative[0])
        """
        WEIGHT UPDATE
        """
        if counter == 1:
            mean_derivative = derivative
        else:
            mean_derivative += derivative

        counter += 1

        if counter == 10:
            mean_derivative = mean_derivative / counter
            theta = theta - lr * mean_derivative
            counter = 1

    print("Loss:", epoch_loss)
    print("------------")
    print(" ")
    epoch_losses.append(epoch_loss.real)


"""
import matplotlib.pyplot as plt
plt.plot([i for i in range(len(epoch_losses))],epoch_losses)
plt.xlabel("Epochs")
plt.ylabel("L2 Loss")
plt.show()
"""

"""
CHECK DERIVATIVES
"""
"""
print("-------------------")
print("DERIVATIVE CHECKING")
print("-------------------")
eps = np.finfo(float).eps ** 3

def forward(image, label, theta, eps):
    x_t = euler_forward(convolve, t0, t1, image, N, theta)
    x = x_t[len(x_t) - 1].astype("complex")
    r_t = relu(x)
    l_t = linear(r_t, theta)
    return L2(l_t, label)

theta = np.concatenate((kernel.reshape(kernel.size,1), matrix.reshape(matrix.size,1)))
theta = theta.astype("complex")
theta[0,0] = complex(theta[0,0],eps)
loss = forward(image, label, theta, eps)
print("Loss is", loss.real)
print("DLdt[0] should be:", loss.imag / eps)

"""
