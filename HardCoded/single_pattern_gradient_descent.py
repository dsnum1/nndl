# gradient descent algorithm:
"""
1. Initialize learning rate
2. Initialize parameters W, b
3. Do the following until convergence:
    1. Compute the output through neuron
    2. Compute loss
    3. Update parameters according to gradients 
"""

import torch
from single_neuron import LinearNeuron
import matplotlib.pyplot as plt




if __name__ == '__main__':
    learning_rate = 0.01
    input_patterns = [
        ([1.0, 0.8], 0.3),
    ]

    n = LinearNeuron(
        w = [1, -2],
        b = -0.2
    )

    loss_graph_data = []
    for x in range (0, 1000):
        input = input_patterns[0][0]
        label = input_patterns[0][1]
        y = n.forward(input)
        J = 0.5*(label-y)*(label-y)
        
        grad_J_u = -(label-y)



        new_weights = []
        for i in range(len(input)):
            new_weights.append(n.w[i] - input[i] *grad_J_u*learning_rate)

        new_bias = -(label-y)*learning_rate
        n = LinearNeuron(
            new_weights,
            new_bias
        )


        loss_graph_data.append((n.w[1], J ))
    


        
    # Extract x and y values from the tuples
    print(n.w)
    print(n.b)
    print(n.forward(input_patterns[0][0]))
    
    data = loss_graph_data
    x_values, y_values = zip(*data)

    # Create a scatter plot
    plt.scatter(x_values, y_values, marker='o', color='b', label='Data Points')

    # Add labels and a legend (if needed)
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Scatter Plot of Data')
    plt.legend()

    # Show the plot
    plt.grid(True)  # Optional: Add grid lines
    plt.show()



        


