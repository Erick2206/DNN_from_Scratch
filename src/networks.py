import numpy as np
import random

class Network:
    def __init__(self, layers):
        self.sizes = layers_size
        self.layers_size = len(layers)
        self.weights = [np.random.randn(r,c) for r,c in zip(layers[:-1], layers[1:])]
        self.bias = [np.random.randn(k,1) for k in sizes]

    def forward(self, input):
        for w,b in zip(self.weights, self.biases):
            input = sigmoid(input*w + b)

    def SGD(self, train_data, epochs, l_rate, mini_batch_size):
        for epoch in range(epochs):
            random.shuffle(train_data)
            mini_batches = [train_data[k:k+mini_batch_size] for k in range(0, len(train_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.MiniBatch(mini_batch, l_rate)

    def MiniBatch(self, mini_batch, l_rate):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for x,y in mini_batch:
            delta_w, delta_b = self.backprop(x,y)
            nabla_w = [w+dw for w,dw in zip(nabla_w,delta_w)]
            nabla_b = [b+db for b,db in zip(nabla_b, delta_b)]

        self.weights = [w - (l_rate/len(mini_batch))*nw for n,nw in zip(self.weights,nabla_w)]
        self.bias = [b - (l_rate/len(mini_batch))*nb for b,nb in zip(self.bias, nabla_b)]

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.bias]

        activation = x
        activations = [x]
        zs = []

        for w,b in zip(self.weights, self.bias):
            z = np.dot(w,activation) + b
            activation = self.sigmoid(z)
            zs.append(activation)

        delta = self.loss_func(activations[-1], y) * self.del_sigmoid(zs[-1])

        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta

        for l in range(2,self.layers_size):
            z = zs[-l]
            sp = self.del_sigmoid(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l+1].transpose())
            nabla_b[-l] = np.dot(delta)

        return (nabla_w, nabla_b)    

    def sigmoid(self, input):
        return 1.0/(1.0+np.exp(-input))

    def del_sigmoid(self, input):
        return sigmoid(input)*(1-sigmoid(z))

    def loss_func(self, y_hat, y):
        return (y_hat-y)
