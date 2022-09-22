# Python-Nueral-net-from-scratch

this project is the result of my first dive into machine learning. I decided to create my own basic feed-forward neural network from scratch (still using numpy for matrix multiplication) so that I could learn how machine learning actually works
I'm also going to be explaining some of the concepts on here to further develeop my understanding of nueral networks. If you notice anything that is incorrect please let me know.

# How it Works

## Forward Propagation 
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

## Backpropagation, how the network learns

Nueral networks use a technique called gradient descent to minimize a cost function. In this case its th squared error of the output (the expected output, $\hat{y}$ minus the actual output $a$, squared (the 1/2 is just for simpler derivation)
$$C = \frac{1}{2}(\hat{y} - a)$$


Gradient descent is the practice of adjusting weights in small steps in order to find the minimum of the cost function. This is done by computing $\frac{\partial{C}}{\partial{w}}$, multiplying it by some learning rate, $\alpha$ and subtracting it from the weights value $w$ 

To compute this gradient we simply use the chain rule. 

first we must compute $\frac{\partial{C}}{\partial{z^L_j}}$ (L denoting the layer and j denoting a row, or nueron). This is also known as the error or delta of a nueron. 
$$\delta^L_j = \frac{\partial{C}}{\partial{z^L_j}} = (a-\hat{y}) * \sigma'(z^L_j)$$
of course, this can be represented in matrix form using a dot product (elementwise multiplication): 
$$\delta^L = \nabla_aC \cdot \sigma'(z^L)$$

note that this formula only applies for the output layer, for the rest of the layers we must compute delta using the following formula:
$$\delta^l = ((w^{l+1})^T\delta^{l+1} \cdot sigma'(z^l)$$

from this we can compute the gradient for both the weights and biases. 
for the biases the gradient is simply: $$\frac{\partial{C}}{b^l_j} = \delta$$
and for the weights the gradient is:
$$\frac{\partial{C}}{\partial{w^l_{jk}}} = a^{l-1}_k\delta^l_j$$
in matrix form this can be represented as:
$$\nabla_wC =(a^{l-1})^T\delta^l$$

finally we can do the "descent" part, to do this we simply multpiply our gradients by our learnin rate $\alpha$ and subtract the value from our weights and biases

$$b = b - \alpha(\delta)$$
$$W = W - \alpha(\delta)$$

# Implementations
## XOR Gate
my first implementation was a model that predicts the output of an xor gate, for this i used a 2-2-1 structure using squared error as my cost function and sigmoid functions for every node. The script to run that implmentation can be found [here](/implementations/XOR/XOR.py)




