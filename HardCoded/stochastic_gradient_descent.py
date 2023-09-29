# stochastic gradient descent algorithm
"""
1. Initialize learning rate
2. Initialize parameters W,b
3. Do until convergence
    For each training patten
        Compute neuron output  
        Calculate loss
        Update W,b accordingly
"""

import torch
from single_neuron import LinearNeuron
import matplotlib.pyplot as plt
from plot_architecture import plot_single_neuron


if __name__ == '__main__':
    learning_rate = 0.001

    input_patterns = [
        ([1.3, 0.9], 0.5),
        ([1.0, -0.2], -1.7),
        ([-0.6, 0.1], -0.7),
        ([-3.27, -0.04], 0.1),
        ([2.0, 1.8], 0.9),
    ]

    n = LinearNeuron(
        w = [1.0, -2.0],
        b = -0.2
    )

    loss_graph_data = []

    for x in range (0, 10000):                   # !FIXME should run it till convergence. 1000 is just an assumed number. But it is not right
        for pattern in input_patterns:
            input = pattern[0]
            label = pattern[1]

            y = n.forward(input)
            print(y)
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


            loss_graph_data.append((n.w[0], n.w[1], n.b, J ))
        


        
    # Extract x and y values from the tuples
    print('=======Final======')
    print('weights: ', n.w)
    print('bias: ', n.b)
    print('output: ',n.forward(input_patterns[0][0]))
    print('=======Final======')
    
    data = loss_graph_data
    w1, w2, b , y_values = zip(*data)

    # Create a scatter plot
    plt.scatter(w1, y_values, marker='o', color='b', label='w1')

        # Add labels and a legend (if needed)
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Scatter Plot of Data')
    plt.legend()

    # Show the plot
    plt.grid(True)  # Optional: Add grid lines
    plt.show()

    plt.scatter(w2, y_values, marker='o', color='b', label='w2')

        # Add labels and a legend (if needed)
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Scatter Plot of Data')
    plt.legend()

    # Show the plot
    plt.grid(True)  # Optional: Add grid lines
    plt.show()

    plt.scatter(b, y_values, marker='o', color='b', label='b')


    # Add labels and a legend (if needed)
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Scatter Plot of Data')
    plt.legend()

    # Show the plot
    plt.grid(True)  # Optional: Add grid lines
    plt.show()



    plot_single_neuron(n, [1.2, 3.5])


    

