import torch

import sys
sys.path.append('D:\\Y4S2\\nndl')

from HardCoded.single_neuron import Neuron
from HardCoded.cost_functions import cross_entropy_loss 

class LogisticRegressionNeuron(Neuron):
    """
    Performs binary classification of inputs. It classifies inputs into one of the two classes: 0 or 1
    Activation of the neuron gives the probability of input1 belonging to either of the two classes.
    """
    def activation_function(self, u):
        y = torch.sigmoid(torch.tensor(u))
        return float(y.item())

    def cost_function(self, y,d):
        return cross_entropy_loss(y,d)
    

