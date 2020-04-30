import numpy as np
from mult import mult
from adj import adjsensitivity
from ode import RKIntegrator
"""
The convolution function

:param image: Input 2D image.
:param theta: Convolutional Network Parameters.
:returns: hl4, the output hiddenlayer of the stacked convolutions.
"""

# From Neural Ordinary Differential Equations, Chen et. al,  2018
def StackedResidualBlock(image, t, theta):
    conv = Conv(None)
    theta = theta.reshape(12,3,3)
    hl1 = conv.forward(relu(conv.forward(image, kernel=theta[0])), kernel=theta[1]) + image
    hl2 = conv.forward(relu(conv.forward(hl1, kernel=theta[2])), kernel=theta[3]) + hl1
    hl3 = conv.forward(relu(conv.forward(hl2, kernel=theta[4])), kernel=theta[5]) + hl2
    hl4 = conv.forward(relu(conv.forward(hl3, kernel=theta[6])), kernel=theta[7]) + hl3
    hl5 = conv.forward(relu(conv.forward(hl4, kernel=theta[8])), kernel=theta[9]) + hl4
    hl6 = conv.forward(relu(conv.forward(hl5, kernel=theta[10])), kernel=theta[11]) + hl5
    return(hl6)

def toyf(image, t, theta):
    conv = Conv(None)
    hl1 = conv.forward(image,kernel=theta[0])
    hl2 = relu(hl1)
    hl3 = conv.forward(image,kernel=theta[0])
    return(hl3)


"""
class StackedResidualBlock:
    def forward(self,A):
         conv = Conv(None)
         theta = theta.reshape(12,3,3)
         self.hl1 = conv.forward(relu(conv.forward(image, kernel=theta[0])), kernel=theta[1]) + image
         self.hl2 = conv.forward(relu(conv.forward(hl1, kernel=theta[2])), kernel=theta[3]) + hl1
         self.hl3 = conv.forward(relu(conv.forward(hl2, kernel=theta[4])), kernel=theta[5]) + hl2
         self.hl4 = conv.forward(relu(conv.forward(hl3, kernel=theta[6])), kernel=theta[7]) + hl3
         self.hl5 = conv.forward(relu(conv.forward(hl4, kernel=theta[8])), kernel=theta[9]) + hl4
         self.hl6 = conv.forward(relu(conv.forward(hl5, kernel=theta[10])), kernel=theta[11]) + hl5
         return(hl6)
     def div(self,output,outputdiv):
         raise NotImplementedError
"""  
class Relu(A):
    def forward(self,A):
        return(np.maximum(A,0))
    def div(self,output,outputdiv):
        return(np.heaviside(output, 0))

def L2(x, label):
    return(np.square(x - label))

"""
class L2:
    def __init__(self, labels):
        self.labels = labels
    def forward(self, x):
        return(np.sum(np.square(x - self.y)))
    def div(self, output, outputdiv):
        return(2*output)
"""

class ConvRKIntegrator:
    def __init__(self, t, theta):
        self.params = theta
        self.t = t
    def forward(self, image):
        return(RKIntegrator(Conv, self.t, image, self.params))
    def div(self, output, outputdiv):
        return(adjsensitivity(Conv, self.params.flatten(), self.t[::-1], output, outputdiv))

class FullyConnected:
    def __init__(self, W, b):
        self.Wshape, self.Wsize = W.shape, W.size
        self.bshape, self.bsize = b.shape, b.size
        self.params = np.concatenate((W.flatten(), b.flatten()))
    def forward(self, image):
        return(mult(image.reshape((1, image.size)), self.params[0:self.Wsize].reshape(self.Wshape)) + self.params[self.Wsize:self.params.size])
    def div(self, output, outputdiv):
        print("FCdiv:", outputdiv)
        return(mult(self.params[0:self.Wsize][:, np.newaxis].reshape(self.Wshape), outputdiv.transpose()).reshape((28,28)))

class Conv:
    def __init__(self, kernel):
        self.params = kernel
    def forward(self, image, stride=1, padding=1, kernel=None):
        self.params = self.params if kernel is None else kernel
        ksize = int(np.sqrt(self.params.size)) # get size of kernel (9x1 array -> ksize = 3)
        hiddenlayer = np.zeros( (int((image.shape[0] - ksize + 2 * padding)/stride + 1),) * 2)
        image = np.pad(image, padding, mode='constant',constant_values=0.0)
        for i in range(0, image.shape[0] - ksize + 1):
            #print("kernel:", kernel.shape)
            #print("image:", image.shape)
            #print("ksize:", ksize, "i:", i)
            #print("hiddenlayer:", hiddenlayer.shape)
            slice = self.unravel(image[i : i + ksize, :])
            #print("slice:", slice.shape)
            #print(" ")
            hiddenlayer[i,:] = mult(self.params.flatten(),slice)
        return hiddenlayer
    def unravel(self, A):
        #print(A.shape)
        unraveled = np.zeros((A.shape[0]**2, A.shape[1] - A.shape[0] + 1))
        #print(unraveled.shape)
        for i in range(A.shape[1] - A.shape[0] + 1):
            unraveled[:,i] = A[0:3,i:i+3].flatten()
        return unraveled
    def div(self, output, outputdiv):
        return(self.forward(image, stride=1, padding=0, kernel=output))
