# Python-Nueral-net-from-scratch

this project is the result of my first dive into machine learning. I decided to create my own basic feed-forward neural network from scratch (still using numpy for matrix multiplication) so that I could learn how machine learning actually works
I'm also going to be explaining some of the concepts on here to further develeop my understanding of nueral networks. If you notice anything that is incorrect please let me know.

## How it Works

### Forward Propagation 
A basic feed-forward Nueral network is comprised of just nodes, sometimes refered to as nuerons, and weights. They also consist of an input layer, an output layer, and a number of hidden layers. The networks weights connect input nodes to the nuerons on the next layer, each input node having its own connection to each neruon. 

<img style="height:400px" src="https://user-images.githubusercontent.com/106884609/191625252-3e7dae53-37f7-4e1d-8842-8cad55164548.png" />
the output (z) of a nueron can be represented as such, where X is the input vector, W is the weight matrix and b is an optional bias: 
$$\sum_{i}^{n}x_iw_i+b_i$$

This summimation can be more simply represented for all nuerons in a layer as a matrix multiplication
<br/>$$z_l = X_lW_l$$

once the weighted sume has been computed, the its fed into an activation function to add a nonlinear effect to the networks output. A few common activation functions include: ReLu, tanh, and sigmoid. 

During the beginning of this project, I have been exclusively working with the sigmoid function as my activation function, especially for my XOR gate implementation
<br/>
$$\sigma (x) = \frac{1}{1+e^-x}$$

$$a_l = \sigma(z_l)$$

### Backpropagation, how the neetwork learns

Nueral networks use a technique called gradient descent to minimize a cost function. In this case its th squared error of the output (the expected output, $\hat{y}$ minus the actual output $a$, squared (the 1/2 is just for simpler derivation)
$$C = \frac{1}{2}(\hat{y} - a)$$


Gradient descent is the practice of adjusting weights in small steps in order to find the minimum of the cost function. This is done by computing the gradient $\frac{\partial{C}}{\partial{w}$

To compute this gradient we simply use the chain rule. 



