# Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

This is the second course of the deep learning specialization at [Coursera](https://www.coursera.org/specializations/deep-learning) which is moderated by [DeepLearning.ai](http://deeplearning.ai/). The course is taught by Andrew Ng.

## Course summary

Here are the course summary as its given on the course [link](https://www.coursera.org/learn/deep-neural-network):

> This course will teach you the "magic" of getting deep learning to work well. Rather than the deep learning process being a black box, you will understand what drives performance, and be able to more systematically get good results. You will also learn TensorFlow. 
>
> After 3 weeks, you will: 
> - Understand industry best-practices for building deep learning applications. 
> - Be able to effectively use the common neural network "tricks", including initialization, L2 and dropout regularization, Batch normalization, gradient checking, 
> - Be able to implement and apply a variety of optimization algorithms, such as mini-batch gradient descent, Momentum, RMSprop and Adam, and check for their convergence. 
> - Understand new best-practices for the deep learning era of how to set up train/dev/test sets and analyze bias/variance
> - Be able to implement a neural network in TensorFlow. 
>
> This is the second course of the Deep Learning Specialization.

## Table of contents

* [Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization](#improving-deep-neural-networks-hyperparameter-tuning-regularization-and-optimization)
   * [Table of contents](#table-of-contents)
   * [Course summary](#course-summary)
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
      * [Deep learning frameworks](#deep-learning-frameworks)
      * [TensorFlow](#tensorflow)
   