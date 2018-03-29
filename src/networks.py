import numpy as np


class Network:
    def __init__(self, layers_size):
        self.sizes = layers_size
        self.weights = np.random.randn()
        self.bias = np.random.randn()

    def forward(self):
        self.sigmoid()

    def MiniBatch(self):
        #TODO

    def SGD(self):
        #TODO

    def backprop(self):
        #TODO

    def sigmoid(self):
        #TODO

    def del_sigmoid(self):
        #TODO            
