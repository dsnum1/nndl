# Batch Gradient Descent
"""
1. Initialize Learning Rate
2. Initialize Parameters W,b
3. Do the following until convergence
    1. Calculate the output
    2. Calculate loss
    3. Update Parameters according to gradients
"""

from single_neuron import LinearNeuron
import matplotlib.pyplot as plt

if __name__ =='__main__':
    print('this is batch gradient descent')
    learning_rate = 0.01
    input = [
        ( -1.6 , 0.2 ),
        ( 1.3 , 2.0 ),
        ( 2.1 , 1.1 ),
        ( -1.0 , -2.8 ),
        ( 1.5 , -2.0 ),
        ( 0.5 , 0.2 ),
        ( 0.2 , -1.9 ),
    ]

    n = LinearNeuron(w=[
        0.2,
    ], b=-1)

    loss_graph_data = []

    for i in range(1000): #!FIXME need to detect convergence
        # Compute output through neuron for all training patterns
        y = []
        for i in range(len(input)):
            y.append(n.forward([input[i][0]]))
        # Compute loss for all training patterns
        loss = 0
        for i in range(len(input)):
            loss+=0.5*(input[i][1]-y[i])*(input[i][1]-y[i])

        print(loss)
        grad_w = 0
        grad_b = 0
        for i in range(len(input)):
            grad_w+= -(input[i][1]-y[i]) * input[i][0]
            grad_b+= -(input[i][1]-y[i])


        n.w = [n.w[0] - learning_rate * grad_w]
        n.b = n.b - learning_rate * grad_b

        loss_graph_data.append((n.w[0], n.b, loss ))
        # Compute gradient for training pattern

        # Update weights and bias according to training pattern

    # input

    # print('=======Final======')
    # print('weights: ', n.w)
    # print('bias: ', n.b)
    # print('output: ',n.forward(input[0][0]))
    # print('=======Final======')
    
    data = loss_graph_data
    w1, b , y_values = zip(*data)

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

    plt.scatter(b, y_values, marker='o', color='b', label='b')


    # Add labels and a legend (if needed)
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Scatter Plot of Data')
    plt.legend()

    # Show the plot
    plt.grid(True)  # Optional: Add grid lines
    plt.show()



    # plot_single_neuron(n, [1.2, 3.5])


    

