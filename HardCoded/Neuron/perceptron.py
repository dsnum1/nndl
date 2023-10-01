import torch


import sys
sys.path.append('D:\\Y4S2\\nndl')
from HardCoded.single_neuron import Neuron
from HardCoded.cost_functions import square_error

class Perceptron(Neuron):
    def activation_function(self, u):
        y = torch.sigmoid(torch.tensor(u))
        return float(y.item())

    def adjusted_activation_function(self, u, amplitude, shift):
        return amplitude * self.activation_function(u) + shift
    
    def forward_adjusted(self, x, amplitude, shift):
        u = self.linear_combination(x)
        # print('this is ',u)
        y = self.adjusted_activation_function(u, amplitude=amplitude, shift=shift)
        return y

    def cost_function(self, d, y):
        return square_error(d,y)
        pass


