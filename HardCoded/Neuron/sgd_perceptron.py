from perceptron import Perceptron
import torch
LR = 0.01

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

def sgd_perceptron(n, input_patterns):
    """
    Algorithm:
    1. Initialize Learning Rate
    2. Initialize W,b
    3. Until Convergence
        1. For each pattern
            1. Calculate output
            2. Calculate loss 
            3. Calculate gradient dJ_w(change in J wrt to w)
            4. Calculate gradient dJ_b(change in J wrt to b)
            5. Update w and b 
    """


    for i in range(0,1000):                        # do the above until convergence
        for pattern in input_patterns:              # iterate through all input patterns
            amplitude = 4.0                         # this is the difference between the minimum value and maximum output values in the input patterns
            shift = -1.0                           # this is the phase shift of the output value
            x = pattern[0]                          # obtain x tuple (x1,x2)
            d = pattern[1]                          # obtain label d
            y = n.forward_adjusted(x, amplitude, shift) # compute output of the graph
            # print(y)
            # print(x, y, d)
            J = n.cost_function(d=d,y=y)
            # print(J)
            dJ_w1 = 0
            dJ_w2 = 0
            dJ_b = 0
            
            activation_derivative = 0.25*(y+1)*(3-y)

            dJ_w1 = -(d-y)* activation_derivative * amplitude *x[0]
            dJ_w2 = -(d-y)* activation_derivative * amplitude *x[1]

            dJ_b = -(d-y)* activation_derivative * amplitude

            w1= n.w[0] - LR * dJ_w1
            w2= n.w[1] - LR * dJ_w2

            b = n.b - LR * dJ_b

            n = Perceptron([w1, w2], b)
            
            # print(J)
            # print(f'Gradients: {dJ_w1, dJ_w2, dJ_b}')
            # print(f'Weights: {n.w}')
            # print(f'Bias: {n.b}')
            print(J)
    
    return n

n = sgd_perceptron(n, input_patterns)
print('=========')
print('n.w: ',n.w)
print('n.b: ',n.b)
print('=========')

for pattern in input_patterns:
    print("input: ", pattern[0], " Output: ", n.forward_adjusted(pattern[0], amplitude=4, shift=-1), " correct: ", pattern[1])

# d = 0.06
# print(n.cost_function(d,y))