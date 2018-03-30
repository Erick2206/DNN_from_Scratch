import numpy as np


class Network:
    def __init__(self, layers):
        self.sizes = layers_size
        self.layers_size = len(layers)
        self.weights = [np.random.randn(r,c) for r,c in zip(layers[:-1], layers[1:])]
        self.bias = [np.random.randn(k,1) for k in sizes]

    def forward(self, input):
        #TODO

    def SGD(self):
        #TODO

    def MiniBatch(self):
        #TODO

    def backprop(self):
        #TODO

    def sigmoid(self):
        #TODO

    def del_sigmoid(self):
        #TODO
