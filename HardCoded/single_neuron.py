import matplotlib.pyplot as plt
import networkx as nx
from plot_architecture import plot_single_neuron
import torch


class Neuron:
    def __init__(self, w, b):
        self.w = w
        self.b = b


    def dot_product(self, list1, list2):
        # Check if the lists have the same length
        if len(list1) != len(list2):
            raise ValueError("Lists must have the same length for dot product calculation.")
        
        result = 0
        for i in range(len(list1)):
            result += list1[i] * list2[i]
        
        return result
    
    def activation_function(self, u):
        return u
    
    def linear_combination(self, x):
        return self.dot_product(x, self.w)+self.b

    def forward(self, x):
        u = self.linear_combination(x)
        y = self.activation_function(u)
        return y
    


class Perceptron(Neuron):
    def activation_function(self, u):
        y = torch.sigmoid(torch.tensor(u))
        return float(y.item())
        return 


class ReLUNeuron(Neuron):
    def activation_function(self, u):
        y = max(0,u)
        return y
    

class LinearNeuron(Neuron):
    pass

class ThresholdNeuron(Neuron):
    def activation_function(self, u):
        y = int(u>0)
        return y

class BipolarSigmoidal(Neuron):
    def activation_function(self, u):
        y =  torch.tanh(torch.tensor(u)).item()
        return y

if __name__ == '__main__':
    # create neuron n
    n = ReLUNeuron(
        w = [
            -1.0,
            0.6,
            0.2,
            0.1,
            -0.6,
            -0.5,
        ],
        b = -0.5
    )


    # create a forward
    input_list = [0.4,0.5,-0.3, 0.4, 0.8, 0.4]
    print(n.forward(input_list))

    plot_single_neuron(n, input_list)

