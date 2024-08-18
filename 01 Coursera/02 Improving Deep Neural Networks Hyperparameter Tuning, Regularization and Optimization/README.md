# Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

This is the second course of the deep learning specialization at [Coursera](https://www.coursera.org/specializations/deep-learning) which is moderated by [DeepLearning.ai](http://deeplearning.ai/). The course is taught by Andrew Ng.

## Course summary

This course offers an in-depth exploration of key principles in deep learning:

- Gain an understanding of industry best practices for developing deep learning applications.
- Master essential neural network techniques, including initialization, L2 and dropout regularization, Batch normalization, and gradient checking.
- Learn how to implement and apply various optimization algorithms like mini-batch gradient descent, Momentum, RMSprop, and Adam, along with methods to ensure their convergence.
- Understand the latest practices in setting up train/dev/test sets and performing bias/variance analysis in the deep learning era.
- Be able to implement neural networks using TensorFlow.

## Table of contents


* [Practical aspects of Deep Learning](#practical-aspects-of-deep-learning)
   * [Train / Dev / Test sets](#train--dev--test-sets)
   * [Bias / Variance](#bias--variance)
   * [Basic Recipe for Machine Learning](#basic-recipe-for-machine-learning)
   * [Regularization](#regularization)
   * [Why regularization reduces overfitting?](#why-regularization-reduces-overfitting)
   * [Dropout Regularization](#dropout-regularization)
   * [Understanding Dropout](#understanding-dropout)
   * [Other regularization methods](#other-regularization-methods)
   * [Normalizing inputs](#normalizing-inputs)
   * [Vanishing / Exploding gradients](#vanishing--exploding-gradients)
   * [Weight Initialization for Deep Networks](#weight-initialization-for-deep-networks)
   * [Numerical approximation of gradients](#numerical-approximation-of-gradients)
   * [Gradient checking implementation notes](#gradient-checking-implementation-notes)
* [Optimization algorithms](#optimization-algorithms)
   * [Mini-batch gradient descent](#mini-batch-gradient-descent)
   * [Understanding mini-batch gradient descent](#understanding-mini-batch-gradient-descent)
   * [Exponentially weighted averages](#exponentially-weighted-averages)
   * [Understanding exponentially weighted averages](#understanding-exponentially-weighted-averages)
   * [Bias correction in exponentially weighted averages](#bias-correction-in-exponentially-weighted-averages)
   * [Gradient descent with momentum](#gradient-descent-with-momentum)
   * [RMSprop](#rmsprop)
   * [Adam optimization algorithm](#adam-optimization-algorithm)
   * [Learning rate decay](#learning-rate-decay)
   * [The problem of local optima](#the-problem-of-local-optima)
* [Hyperparameter tuning, Batch Normalization and Programming Frameworks](#hyperparameter-tuning-batch-normalization-and-programming-frameworks)
   * [Tuning process](#tuning-process)
   * [Using an appropriate scale to pick hyperparameters](#using-an-appropriate-scale-to-pick-hyperparameters)
   * [Hyperparameters tuning in practice: Pandas vs. Caviar](#hyperparameters-tuning-in-practice-pandas-vs-caviar)
   * [Normalizing activations in a network](#normalizing-activations-in-a-network)
   * [Fitting Batch Normalization into a neural network](#fitting-batch-normalization-into-a-neural-network)
   * [Why does Batch normalization work](#why-does-batch-normalization-work)
   * [Batch normalization at test time](#batch-normalization-at-test-time)
   * [Softmax Regression](#softmax-regression)
   * [Training a Softmax classifier](#training-a-softmax-classifier)
