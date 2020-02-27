sssimport numpy as np
from mult import mult
from adj import adjsensitivity

"""
The convolution function

:param image: Input 2D image.
:param theta: Convolutional Network Parameters.
:returns: hl4, the output hiddenlayer of the stacked convolutions.
"""
def convolution(image, theta):
    hl1 = nn.conv(image,theta[0])
    hl2 = nn.conv(image,theta[1])
    hl3 = nn.conv(image,theta[2])
    hl4 = nn.conv(image,theta[3])
    return(hl4)

class L2:
    def forward(self, x, y):
        return(np.sum(np.square(x - y)))
    def div(self, output, outputdiv):
        return(2*output)

class ConvRKIntegrator:
    def __init__(self, theta, t, convolution):
        self.params = theta.flatten()
        self.t = t
    def forward(self, image):
        return(KIntegrator(convolution(image, self.params), self.t, image, self.params))
    def div(self, output, outputdiv):
        return(adjsensitivity(convolution, self.params, t[0], t[len(t) - 2], output), previous)

class fc:
    def __init__(self, W, b):
        self.Wshape, self.Wsize = shape(W), size(W)
        self.bshape, self.bsize = shape(b), size(b)
        self.params = np.concatenate(W.flatten(), b.flatten())
    def forward(self, image):
        return(mult(self.params[0:self.Wsize].reshape(self.Wshape), image.ravel()) + self.params[self.Wsize:])
    def div(self, output, outputdiv):
        return(outputdiv * W[:, np.newaxis])

class conv:
    def __init__(self, kernel):
        self.params = kernel
    def forward(self, image, stride=1, padding=1):
        ksize = int(np.sqrt(max(self.params.shape))) # get size of kernel (9x1 array -> ksize = 3)
        hiddenlayer = np.zeros( (int((image.shape[0] - ksize + 2 * padding)/stride + 1),) * 2)
        image = np.pad(image, padding, mode='constant',constant_values=0.0)
        for i in range(ksize, image.shape[0] - ksize):
            slice = self.unravel(image[i - ksize : i, :])
            hiddenlayer[i,:] = mult(self.params,slice)
            return hiddenlayer
    def unravel(self, A):
        unraveled = np.zeros((A.shape[0]**2, A.shape[1] - A.shape[0] + 1))
        for i in range(A.shape[1] - A.shape[0] + 1):
            unraveled[:,i] = A[0:3,i:i+3].flatten()
        return unraveled
    def div(self, output, outputdiv):
        raise NotImplementedError
