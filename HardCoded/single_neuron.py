import matplotlib.pyplot as plt
import networkx as nx
from plot_architecture import plot_single_neuron

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
    

if __name__ == '__main__':
    # create neuron n
    n = Neuron(
        w = [
            1.0,
            -2.6,
            -1.5
        ],

        b = -0.5
    )


    # create a forward
    print(n.forward([1,2,3]))

    plot_single_neuron(n, [1,2,3])

