# Neural Network and Deep Learning
This is a self-exploration of artificial neural networks.

## Chapters
- Introduction
- Single Neuron




## Single Neuron
HardCoded/single_neuron.py contains a parent class that implements a single neuron. 
A neuron is a fundamental element of all neural nets. A typical neuron takes a weighted sum of all its input signals. This sum is a single number that is input into an activation function. The activation function computes the final output. In the code provided, you can define a neural net using the Neuron class.
``` 
n = Neuron(w = [a,b,c,..], b = x )
```

w means weights. It is a list which stores the weight that every ith input will be multiplied with 


### Perceptron
A perceptron is a neuron with a sigmoidal activation function. They are used to perform non-linear regression of inputs.
The square error function is used as the cost function for this perceptron. 
```
J = 0.5(y-d)^2
```

#### SGD for Perceptron
  __Algorithm__:
  1. Initialize Learning Rate
  2. Initialize W,b
  3. Until Convergence
      1. For each pattern
          1. Calculate output.
          ```
          y = n.forward(pattern[0])
          ```
          2. Calculate loss 
          ```
          d = pattern[1]
          J = n.cost_function(d=d,y=y)
          ```
          3. Calculate gradient J_w(change in J wrt to w)
          Applying the Chain rule. ∇J_w = ∇J_y * ∇y_u * ∇u_w
          - ∇J_y = -(d-y)
          - ∇y_u = f'(u) = f(u)*(1-f(u))
          - ∇u_w = x
          ```
          dJ_w = -(d-y)*torch.sigmoid(u).item() * (1-torch.sigmoid(u).item()) *x
          ```
          4. Calculate gradient J_b(change in J wrt to b)
          Applying the Chain rule. ∇J_b = ∇J_y * ∇y_u * ∇u_b
          - ∇J_y = -(d-y)
          - ∇y_u = f'(u) = f(u)*(1-f(u))
          - ∇u_b = 1
          ```
          dJ_b = -(d-y)*torch.sigmoid(u).item() * (1-torch.sigmoid(u).item())
          ```
            5. Update w and b 
            ```
            w = w - learning_rate * dJ_w
            b = b - learning_rate * dJ_b
            ```



