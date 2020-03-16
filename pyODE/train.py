import nn
from copy import deepcopy
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy.random import rand
from time import time

# FUNCTION DEFINITIONS: ALL HIGH LEVEL BUILDING BLOCKS OF TRAINING ALGORITMS

# NeuralNet() is the forward function, and is essentially d/dx in this problem.
def NeuralNet(image, layers):
    states = [deepcopy(image)]
    for i,layer in enumerate(layers):
        states.append(layer.forward(states[i]))
    return(states)

# autodiff() constructs the differentiation "graph" (right now it's just a line)
# of derivatives based on the layer list.
def autodiff(states, layers, loss):
    derivatives = []
    for i, (layer,state) in enumerate(zip(reversed(layers), reversed(states[1:]))):
        derivatives = layer.div(state, derivatives[i-1] if i!=0 else loss)
    return(derivatives)

# Backpropagate() takes the derivatives and accesses their parameters and
# adjusts them according to the derivatives computed in autodiff().
def backpropagate(layers, derivatives, learningrate):
    for layer,derivative in zip(layers,derivatives.reverse()):
        layer.params = layer.params - learningrate * derivative
    return(layers)

# train() takes in the images and labels and returns the final parameters of the
# neural net.
def train(images, labels, layers):
    for image,label in zip(images,labels):
        states = NeuralNet(image, layers)
        loss = nn.L2(states[len(states) - 1], labels)
        derivatives = autodiff(states, layers, loss)
        layers = backpropapagate(layers, derivatives, learningrate)
        print("image finished")
    return(theta)

# data() reads in MNIST from my disk. This code is bad, and should not be published
# or shown to anyone, but damn if it isn't slick. It could be even less verbose, but
# it goes on GitHub.
def data(path):
    start = time() # Timing function
    images, groundtruths = [], []
    for file in os.listdir(path): #Iterate through train folder
        images.append(np.asarray(Image.open(path + file)) / 255) #Load image into memory
        groundtruths.append(np.asarray((int(file[10]))*[0] + [1] + (9 - int(file[10]))*[0])) #literally do not ever do this
        break #REMOVE LATER, ONLY LOADS ONE IMAGE
    print("Data Loaded... ", len(images), "images loaded, ", round(time() - start, 2), "s elapsed.")
    return(images, groundtruths)

images, labels = data("../RKNet/data/train/")
layers = [nn.ConvRKIntegrator(np.linspace(0,2,3), rand(12,3,3)), nn.FullyConnected(rand(28*28,10), rand(10,1))]
params = train(images, labels, layers)



#print(mult.mult.calls, "Matrix Multiplication Operations")
