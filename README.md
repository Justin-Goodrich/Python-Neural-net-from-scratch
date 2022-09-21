# Python-Nueral-net-from-scratch

this project is the result of my first dive into machine learning. I decided to create my own basic feed-forward neural network from scratch (still using numpy for matrix multiplication) so that I could learn how machine learning actually works
I'm also going to be explaining some of the concepts on here to further develeop my understanding of nueral networks. If you notice anything that is incorrect please let me know.

## How it Works

A basic feed-forward Nueral network is comprised of just nodes, sometimes refered to as nuerons, and weights. They also consist of an input layer, an output layer, and a number of hidden layers. The networks weights connect input nodes to the nuerons on the next layer, each input node having its own connection to each neruon. 

the output (z) of a nueron can be represented as such, where X is the input vector, W is the weight matrix and b is an optional bias: 
$$\sum_{i}^{n}x_iw_i+b_i$$

This summimation can be more simply represented for all nuerons in a layer as a matrix multiplication
<br/>$z_l = X_lW_l$

once the weighted sume has been computed, the its fed into an activation function to add a nonlinear effect to the networks output. A few common activation functions include: ReLu, tanh, and sigmoid. 

During the beginning of this project, I have been exclusively working with the sigmoid function as my activation function, especially for my XOR gate implementation
<br/>
$$\sigma (x) = \frac{1}{1+e^-x}$$

$$a_l = \sigma(z_l)$$







