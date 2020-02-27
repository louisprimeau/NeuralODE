import nn
from loss import L2
from derivative import L2div, FCdiv
from copy import deepcopy

def NeuralNet(image, theta, layers):
    states = [image]
    for i,layer in enumerate(layers):
        states.append(layer.forward(states[i]))
    return(state)

def autodiff(states, layers):
    derivatives = []
    for i, (layer,state) in enumerate(zip(layers.reverse(), states[1:].reverse())):
        derivatives = layer.div(state, derivatives[i-1] if i==0 else None)
    return(derivatives)

def backpropagate(layers, derivatives, learningrate):
    for layer,derivative in zip(layers,derivatives.reverse()):
        layer.params = layer.params - learningrate * derivative
    return(layers)

def train(images, labels, layers):
    for image,label in zip(images,labels):
        states = NeuralNet(image, theta, layers)
        derivatives = autodiff(states, layers)
        layers = backpropapagate(layers, derivatives, learningrate)
        print("image finished")
    return(theta)
