# Neural Networks and Deep Learning

This course is the first in the Deep Learning specialization offered by [Coursera](https://www.coursera.org/specializations/deep-learning) and moderated by [DeepLearning.ai](http://deeplearning.ai). The course is taught by Andrew Ng.

## Course Summary

This course serves as an introduction to deep learning. You will learn:
- The major technology trends driving Deep Learning.
- How to build, train, and apply fully connected deep neural networks.
- How to implement efficient (vectorized) neural networks.
- Key parameters in a neural network's architecture.

By the end of the course, you'll be able to apply deep learning techniques to your own projects and be prepared for AI-related job interviews.

## Table of Contents

* [Neural Networks and Deep Learning](#neural-networks-and-deep-learning)
   * [Table of contents](#table-of-contents)
   * [Course summary](#course-summary)
   * [Introduction to deep learning](#introduction-to-deep-learning)
      * [What is a (Neural Network) NN?](#what-is-a-neural-network-nn)
      * [Supervised learning with neural networks](#supervised-learning-with-neural-networks)
      * [Why is deep learning taking off?](#why-is-deep-learning-taking-off)
   * [Neural Networks Basics](#neural-networks-basics)
      * [Binary classification](#binary-classification)
      * [Logistic regression](#logistic-regression)
      * [Logistic regression cost function](#logistic-regression-cost-function)
      * [Gradient Descent](#gradient-descent)
      * [Derivatives](#derivatives)
      * [More Derivatives examples](#more-derivatives-examples)
      * [Computation graph](#computation-graph)
      * [Derivatives with a Computation Graph](#derivatives-with-a-computation-graph)
      * [Logistic Regression Gradient Descent](#logistic-regression-gradient-descent)
      * [Gradient Descent on m Examples](#gradient-descent-on-m-examples)
      * [Vectorization](#vectorization)
      * [Vectorizing Logistic Regression](#vectorizing-logistic-regression)
      * [Notes on Python and NumPy](#notes-on-python-and-numpy)
      * [General Notes](#general-notes)
   * [Shallow neural networks](#shallow-neural-networks)
      * [Neural Networks Overview](#neural-networks-overview)
      * [Neural Network Representation](#neural-network-representation)
      * [Computing a Neural Network's Output](#computing-a-neural-networks-output)
      * [Vectorizing across multiple examples](#vectorizing-across-multiple-examples)
      * [Activation functions](#activation-functions)
      * [Why do you need non-linear activation functions?](#why-do-you-need-non-linear-activation-functions)
      * [Derivatives of activation functions](#derivatives-of-activation-functions)
      * [Gradient descent for Neural Networks](#gradient-descent-for-neural-networks)
      * [Random Initialization](#random-initialization)
   * [Deep Neural Networks](#deep-neural-networks)
      * [Deep L-layer neural network](#deep-l-layer-neural-network)
      * [Forward Propagation in a Deep Network](#forward-propagation-in-a-deep-network)
      * [Getting your matrix dimensions right](#getting-your-matrix-dimensions-right)
      * [Why deep representations?](#why-deep-representations)
      * [Building blocks of deep neural networks](#building-blocks-of-deep-neural-networks)
      * [Forward and Backward Propagation](#forward-and-backward-propagation)
      * [Parameters vs Hyperparameters](#parameters-vs-hyperparameters)
      * [What does this have to do with the brain](#what-does-this-have-to-do-with-the-brain)
   

## Introduction to deep learning

> Understand the major trends driving the rise of deep learning and how it is applied today.

### What is a (Neural Network) NN?

- A single neuron in a neural network can be viewed as a linear regression model.
- A simple NN consists of layers of neurons, with the most popular activation function being ReLU (Rectified Linear Unit), which enables faster training.
- Hidden layers in deep learning automatically predict connections between inputs, forming the essence of deep learning.
- A deep NN contains multiple hidden layers, allowing it to model complex patterns.
- Supervised learning involves mapping inputs (X) to outputs (Y) using a neural network.

### Supervised learning with neural networks

- Different neural networks are specialized for different tasks:
  - CNNs (Convolutional Neural Networks) are effective for computer vision.
  - RNNs (Recurrent Neural Networks) are used for sequence-based data like speech or text.
  - Standard NNs are applied to structured data.
  - Hybrid or custom NNs combine different types.
- Structured data refers to data organized in tables, while unstructured data includes images, video, audio, and text.
- Structured data is often more profitable as companies use it to make data-driven predictions.

### Why is deep learning taking off?

- Deep learning has become prominent due to three main factors:
  1. **Data**: With increasing data availability, neural networks outperform traditional methods like SVM.
  2. **Computation**: The advent of GPUs, powerful CPUs, distributed computing, and ASICs has accelerated deep learning.
  3. **Algorithms**: New algorithms, such as using ReLU over Sigmoid, have resolved issues like the vanishing gradient problem.

## Neural Networks Basics

> Set up a machine learning problem with a neural network mindset and use vectorization to optimize models.

### Binary classification

- Logistic regression is used for binary classification, e.g., determining if an image contains a cat.
- Key notations:
  - `M`: Number of training examples
  - `Nx`: Size of the input vector
  - `Ny`: Size of the output vector
  - `X(1)`: First input vector
  - `Y(1)`: First output vector

### Logistic regression

- Logistic regression equations:
  - `y = wx + b` (Simple equation)
  - `y = sigmoid(w(transpose)x + b)` for probabilities between 0 and 1.
- In binary classification, `Y` must be between 0 and 1.

### Logistic regression cost function

- The square root error function is not used due to non-convexity, leading to optimization issues.
- The chosen function is `L(y',y) = - (y*log(y') + (1-y)*log(1-y'))`.
- The cost function averages the loss functions across the entire training set.

### Gradient Descent

- Gradient descent optimizes `w` and `b` to minimize the cost function, which is convex.
- The algorithm updates parameters using the learning rate and derivatives:
  - `w = w - alpha * d(J(w,b) / dw)`
  - `b = b - alpha * d(J(w,b) / db)`

### Derivatives

- Derivatives represent the slope of a function and are essential in optimizing neural networks.
- Example derivatives:
  - For `f(a) = a^2`, the derivative `d(f(a))/d(a) = 2a`.

### More Derivatives examples

- Derivatives of common functions:
  - `f(a) = a^3` → `d(f(a))/d(a) = 3a^2`
  - `f(a) = log(a)` → `d(f(a))/d(a) = 1/a`

### Computation graph

- A computation graph organizes computations from left to right, facilitating the calculation of derivatives using the chain rule.

### Derivatives with a Computation Graph

- Derivatives on a graph are computed from right to left.
- Example: `dvar` denotes derivatives of the output variable with respect to intermediate quantities.

### Logistic Regression Gradient Descent

- The logistic regression pseudo code uses both forward and backward passes to optimize weights and biases.

### Gradient Descent on m Examples

- The logistic regression code can be vectorized to eliminate loops and optimize computation.

### Vectorization

- Vectorization in deep learning reduces the need for loops, speeding up computations.
- NumPy's dot function is inherently vectorized, and broadcasting handles mismatched matrix shapes.

### Vectorizing Logistic Regression

- Implementing logistic regression with vectorization involves replacing loops with matrix operations.

### Notes on Python and NumPy

- NumPy's `sum(axis=0)` and `sum(axis=1)` operations sum columns and rows, respectively.
- Broadcasting in NumPy ensures matrix operations proceed even when shapes don't initially match.
- Reshaping and assertions help prevent bugs in deep learning code.

### General Notes

- Steps to build a Neural Network:
  1. Define the model structure (input features and outputs).
  2. Initialize model parameters.
  3. Loop:
     - Calculate loss (forward propagation).
     - Calculate gradient (backward propagation).
     - Update parameters (gradient descent).
- Hyperparameter tuning, such as learning rate, significantly impacts performance.



## Shallow neural networks

> Build a neural network with one hidden layer using forward propagation and backpropagation.

### Neural Networks Overview

- In a logistic regression model:
  - `X1`, `X2`, `X3` → `z = XW + B` → `a = Sigmoid(z)` → `l(a,Y)`
- In a neural network with one hidden layer:
  - `X1`, `X2`, `X3` → `z1 = XW1 + B1` → `a1 = Sigmoid(z1)` → `z2 = a1W2 + B2` → `a2 = Sigmoid(z2)` → `l(a2,Y)`

### Neural Network Representation

- A neural network consists of input layers, hidden layers, and output layers.
- Hidden layers are not visible in the training set.
- The number of layers in a network determines whether it is shallow or deep.

### Computing a Neural Network's Output

- The forward propagation equations:
  - `Z1 = W1 * X + B1`
  - `A1 = Sigmoid(Z1)`
  - `Z2 = W2 * A1 + B2`
  - `A2 = Sigmoid(Z2)`
- These equations are vectorized across multiple examples.

### Vectorizing across multiple examples

- Vectorized pseudo code for forward propagation:
  - `Z1 = W1 * X + B1`
  - `A1 = Sigmoid(Z1)`
  - `Z2 = W2 * A1 + B2`
  - `A2 = Sigmoid(Z2)`

### Activation functions

- Sigmoid and Tanh functions are commonly used activation functions.
- ReLU is preferred for faster convergence in deep networks.
- Activation functions should be non-linear to allow for complex pattern recognition.

### Why do you need non-linear activation functions?

- Non-linear activation functions enable neural networks to model complex relationships.
- Without non-linearities, the network would behave like a linear regression model, regardless of depth.

### Derivatives of activation functions

- Derivatives of activation functions are used in backpropagation to update weights.
  - Sigmoid: `g'(z) = g(z) * (1 - g(z))`
  - Tanh: `g'(z) = 1 - g(z)^2`
  - ReLU: `g'(z) = 1` for `z >= 0`, `0` for `z < 0`

### Gradient descent for Neural Networks

- Gradient descent is applied across layers using backpropagation to optimize the network.
- The equations for backpropagation update the weights and biases in the network.

### Random Initialization

- Random initialization of weights breaks symmetry, allowing the network to learn effectively.
- Small random values are used to avoid saturation in activation functions like Sigmoid or Tanh.

## Deep Neural Networks

> Understand the key computations underlying deep learning, use them to build and train deep neural networks, and apply it to computer vision.

### Deep L-layer neural network

- A deep neural network contains multiple hidden layers (`L` denotes the number of layers).
- The structure of the network, including the number of neurons and activation functions, is critical.

### Forward Propagation in a Deep Network

- Forward propagation involves computing activations for each layer using the weights and biases.
- Each layer's output serves as the input for the next layer.

### Getting your matrix dimensions right

- Correct matrix dimensions are essential to avoid errors during computation.
- The dimensions must align according to the network's architecture.

### Why deep representations?

- Deep networks model hierarchical patterns, from simple to complex.
- Example: Face recognition models learn edges, then facial features, and finally entire faces.

### Building blocks of deep neural networks

- Forward and backward propagation are the core components of training a deep neural network.
- Each layer's parameters are updated through backpropagation.

### Forward and Backward Propagation

- Forward propagation computes the output for each layer.
- Backward propagation updates the parameters based on the gradient of the loss function.

### Parameters vs Hyperparameters

- **Parameters**: Weights (`W`) and biases (`b`) that the model learns.
- **Hyperparameters**: Values like learning rate, number of layers, and activation functions that need to be tuned.

### What does this have to do with the brain

- Neural networks are loosely inspired by the human brain, but the analogy is limited.
- The most accurate brain-inspired models are used in computer vision (CNNs).


