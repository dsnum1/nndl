from perceptron import Perceptron
import torch
LR = 0.001


n = Perceptron(
    [0.81, 0.61],
    0.0
)

input_patterns = [
    ([0.77, 0.02], 2.91),
    ([0.63, 0.75], 0.55),
    ([0.50, 0.22], 1.28),
    ([0.20, 0.76], -0.74),
    ([0.17, 0.09], 0.88),
    ([0.69, 0.95], 0.30),
    ([0.00, 0.51], -0.28),
]


def gd_perceptron(n, input_patterns):
    """
    Algorithm:
    1. Initialize learning rate
    2. Initialize w,b
    3. Until convergence
        1. Calculate output of all training patterns
        2. Calculate error for each training pattern and sum it to calculate total loss
        3. Calculate change in d
    
    """

    for i in range(1000): # until convergence
        



