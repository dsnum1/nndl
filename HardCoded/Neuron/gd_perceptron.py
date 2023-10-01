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
        3. Calculate change in dJ_w(change in J wrt to w)
        4. Calculate change in dJ_b(change in J wrt to b)
        5. Update w and b
    """
    amplitude = 4
    shift = -1
    for i in range(1000): # until convergence


        # calculate the output and error
        # y_list = []
        # J_list = []
        dJ_w0 = 0
        dJ_w1 = 0
        dJ_b = 0
        for pattern  in input_patterns:
            x = pattern[0]
            d = pattern[1]
            y = n.forward_adjusted(x, amplitude, shift) # compute output of the graph
            J = n.cost_function(d=d, y=y)
            # J_list.append(J)
            # y_list.append(y)
            activation_derivative = 0.25*(y+1)*(3-y)

            dJ_w0 -= (d-y)* activation_derivative * amplitude* x[0]
            dJ_w1 -= (d-y)* activation_derivative * amplitude* x[1]
            dJ_b  -= (d-y)* activation_derivative * amplitude


        
        w1 = n.w[0] - LR * dJ_w0
        w2 = n.w[1] - LR * dJ_w1
        b = n.b - LR * dJ_b

        n = Perceptron([w1,w2], b)
        #calculate error
    return n




        


n = gd_perceptron(n, input_patterns)
print('=========')
print('n.w: ',n.w)
print('n.b: ',n.b)
print('=========')

for pattern in input_patterns:
    print("input: ", pattern[0], " Output: ", n.forward_adjusted(pattern[0], amplitude=4, shift=-1), " correct: ", pattern[1])

# d = 0.06
# print(n.cost_function(d,y))
