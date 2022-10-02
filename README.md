# Python-Nueral-net-from-scratch

This project is the result of my first dive into machine learning. I decided to create my own basic feed-forward neural network from scratch (still using NumPy for matrix multiplication) so that I could learn how machine learning actually works I'm also going to be explaining some of the concepts here to further develop my understanding of neural networks. If you notice anything that is incorrect please let me know.

# How it Works

## Forward Propagation 
A basic feed-forward Neural network is comprised of just nodes, sometimes referred to as neurons, and weights. They also consist of an input layer, an output layer, and a number of hidden layers. The network weights connect input nodes to the neurons on the next layer, each input node having its own connection to each neuron.

<img style="height:400px" src="https://user-images.githubusercontent.com/106884609/191625252-3e7dae53-37f7-4e1d-8842-8cad55164548.png" />
The output (z) of a neuron can be represented as such, where X is the input vector, W is the weight matrix and b is an optional bias:
$$\sum_{i}^{n}x_iw_i+b_i$$

This summation can be more simply represented for all neurons in a layer as a matrix multiplication:
<br/>$$z_l = X_lW_l$$
once the weighted sum has been computed, its fed into an activation function to add a nonlinear effect to the network's output. A few common activation functions include ReLu, tanh, and sigmoid.

During the beginning of this project, I have been exclusively working with the sigmoid function as my activation function, especially for my XOR gate implementation.
<br/>
$$\sigma (x) = \frac{1}{1+e^-x}$$

$$a_l = \sigma(z_l)$$

## Backpropagation, how the network learns

Neural networks use a technique called gradient descent to minimize a cost function. In this case, its the squared error of the output (the expected output,$\hat{y}$ minus the actual output $a$, squared (the 1/2 is just for simpler derivation)
$$C = \frac{1}{2}(\hat{y} - a)$$


Gradient descent is the practice of adjusting weights in small steps in order to find the minimum of the cost function. This is done by computing $\frac{\partial{C}}{\partial{w}}$, multiplying it by some learning rate, $\alpha$ and and subtracting it from the weights value $w$ 

To compute this gradient we simply use the chain rule.

first we must compute $\frac{\partial{C}}{\partial{z^L_j}}$ (L denoting the layer and j denoting a row, or neuron). This is also known as the error or delta of a neuron.
$$\delta^L_j = \frac{\partial{C}}{\partial{z^L_j}} = (a-\hat{y}) * \sigma'(z^L_j)$$
of course, this can be represented in matrix form using a dot product (elementwise multiplication):
$$\delta^L = \nabla_aC \cdot \sigma'(z^L)$$

note that this formula only applies to the output layer, for the rest of the layers we must compute the delta using the following formula:
$$\delta^l = ((w^{l+1})^T\delta^{l+1} \cdot \sigma'(z^l)$$

from this, we can compute the gradient for both the weights and biases. for the biases, the gradient is simply:
$$\frac{\partial{C}}{b^l_j} = \delta$$
and for the weights the gradient is:
$$\frac{\partial{C}}{\partial{w^l_{jk}}} = a^{l-1}_k\delta^l_j$$
in matrix form this can be represented as:
$$\nabla_wC =(a^{l-1})^T\delta^l$$

finally, we can do the "descent" part, to do this we simply multiply our gradients by our learning rate $\alpha$ and subtract the value from our weights and biases

$$b = b - \alpha(\delta)$$
$$W = W - \alpha(\delta)$$

# Implementations
## XOR Gate
My first implementation was a model that predicts the output of an xor gate, for this I used a 2-2-1 structure using squared error as my cost function and sigmoid functions for every node. The script to run that implementation can be found [here](/implementations/XOR/XOR.py).

## Classifying Handwritten Digits
This implementation classifies handwritten digits from the well-known, MNIST dataset. It simply flattens the image matrix and feeds it into the network, which contains two hidden layers. The structure is 728 (flattened pixel matrix) - 128 - 64 - 10 (output vector). The input layer as well as all of the hidden layers utilize the sigmoid activation function, but the output layer uses the softmax activation function to show output in terms of probabilities. The MNIST implmentation script can be found  [here](/implementations/MNIST_digits/mnist.py).
