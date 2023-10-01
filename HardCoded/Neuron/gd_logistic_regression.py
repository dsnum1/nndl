from logistic_regression import LogisticRegressionNeuron
LR = 0.04

n = LogisticRegressionNeuron(
    [0.77, 0.2],
    0.0
)


input_patterns = [
    ([1.33, 0.72], 1),
    ([-1.55, -0.01], 0),
    ([0.62, -0.72], 0),
    ([0.27, 0.11], 0),
    ([0.00, -0.17], 0),
    ([0.43, 1.20], 1),
    ([-0.97, 1.03], 1),
    ([0.23, 0.45], 1)
]

def gd_logistic_neuron(n, input_patterns):
    for i in range(1000):
        dJ_w1 = 0
        dJ_w2 = 0
        dJ_b  = 0
        for pattern in input_patterns:
            x = pattern[0]
            d = pattern[1]
            y = n.forward(x)
            J = n.cost_function(y,d)

            dJ_w1+= (-(d - y))*x[0] 
            dJ_w2+= (-(d - y))*x[1] 
            dJ_b += (-(d - y))
        

        w1 = n.w[0] - LR * dJ_w1
        w2 = n.w[1] - LR * dJ_w2 
        b  = n.b  - LR *dJ_b

        n = LogisticRegressionNeuron([w1,w2], b)

    return n

n = gd_logistic_neuron(n, input_patterns)
print('=========')
print('n.w: ',n.w)
print('n.b: ',n.b)
print('=========')

for pattern in input_patterns:
    print("input: ", pattern[0], " Output: ", n.forward(pattern[0]), " correct: ", pattern[1])

