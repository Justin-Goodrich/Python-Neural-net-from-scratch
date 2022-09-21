# Python-Nueral-net-from-scratch

this project is the result of my first dive into machine learning. I decided to create my own basic feed-forward neural network from scratch (still using numpy for matrix multiplication) so that I could learn how machine learning actually works
I'm also going to be explaining some of the concepts on here to further develeop my understanding of nueral networks.

## How it Works

A basic feed-forward Nueral network is comprised of just nodes, sometimes refered to as nuerons, and weights. They also consist of an input layer, an output layer, and a number of hidden layers. The networks weights connect input nodes to the nuerons on the next layer, each input node having its own connection to each neruon. 

the output of a nueron can be represented as such, where x is the input, w is the weight and b is an optional bias node: 
$$\sum_{i}^{n}x_iw_i+b_i$$

This summimation can be more simply represented for all nuerons in a layer as a matrix multiplication
<br/>$O^l = X^lw^l$


