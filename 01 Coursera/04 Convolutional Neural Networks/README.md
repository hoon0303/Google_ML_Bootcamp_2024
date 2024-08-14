# Convolutional Neural Networks

This is the forth course of the deep learning specialization at [Coursera](https://www.coursera.org/specializations/deep-learning) which is moderated by [DeepLearning.ai](http://deeplearning.ai/). The course is taught by Andrew Ng.

## Course summary

Here is the course summary as given on the course [link](https://www.coursera.org/learn/convolutional-neural-networks):

> This course will teach you how to build convolutional neural networks and apply it to image data. Thanks to deep learning, computer vision is working far better than just two years ago, and this is enabling numerous exciting applications ranging from safe autonomous driving, to accurate face recognition, to automatic reading of radiology images. 
>
> You will:
> - Understand how to build a convolutional neural network, including recent variations such as residual networks.
> - Know how to apply convolutional networks to visual detection and recognition tasks.
> - Know to use neural style transfer to generate art.
> - Be able to apply these algorithms to a variety of image, video, and other 2D or 3D data.
>
> This is the fourth course of the Deep Learning Specialization.

## Table of contents

* [Convolutional Neural Networks](#convolutional-neural-networks)
   * [Table of contents](#table-of-contents)
   * [Course summary](#course-summary)
   * [Foundations of CNNs](#foundations-of-cnns)
      * [Computer vision](#computer-vision)
      * [Edge detection example](#edge-detection-example)
      * [Padding](#padding)
      * [Strided convolution](#strided-convolution)
      * [Convolutions over volumes](#convolutions-over-volumes)
      * [One Layer of a Convolutional Network](#one-layer-of-a-convolutional-network)
      * [A simple convolution network example](#a-simple-convolution-network-example)
      * [Pooling layers](#pooling-layers)
      * [Convolutional neural network example](#convolutional-neural-network-example)
      * [Why convolutions?](#why-convolutions)
   * [Deep convolutional models: case studies](#deep-convolutional-models-case-studies)
      * [Why look at case studies?](#why-look-at-case-studies)
      * [Classic networks](#classic-networks)
      * [Residual Networks (ResNets)](#residual-networks-resnets)
      * [Why ResNets work](#why-resnets-work)
      * [Network in Network and 1×1 convolutions](#network-in-network-and-1-X-1-convolutions)
      * [Inception network motivation](#inception-network-motivation)
      * [Inception network (GoogleNet)](#inception-network-googlenet)
      * [Using Open-Source Implementation](#using-open-source-implementation)
      * [Transfer Learning](#transfer-learning)
      * [Data Augmentation](#data-augmentation)
      * [State of Computer Vision](#state-of-computer-vision)
   * [Object detection](#object-detection)
      * [Object Localization](#object-localization)
      * [Landmark Detection](#landmark-detection)
      * [Object Detection](#object-detection-1)
      * [Convolutional Implementation of Sliding Windows](#convolutional-implementation-of-sliding-windows)
      * [Bounding Box Predictions](#bounding-box-predictions)
      * [Intersection Over Union](#intersection-over-union)
      * [Non-max Suppression](#non-max-suppression)
      * [Anchor Boxes](#anchor-boxes)
      * [YOLO Algorithm](#yolo-algorithm)
      * [Region Proposals (R-CNN)](#region-proposals-r-cnn)
   * [Special applications: Face recognition &amp; Neural style transfer](#special-applications-face-recognition--neural-style-transfer)
      * [Face Recognition](#face-recognition)
         * [What is face recognition?](#what-is-face-recognition)
         * [One Shot Learning](#one-shot-learning)
         * [Siamese Network](#siamese-network)
         * [Triplet Loss](#triplet-loss)
         * [Face Verification and Binary Classification](#face-verification-and-binary-classification)
      * [Neural Style Transfer](#neural-style-transfer)
         * [What is neural style transfer?](#what-is-neural-style-transfer)
         * [What are deep ConvNets learning?](#what-are-deep-convnets-learning)
         * [Cost Function](#cost-function)
         * [Content Cost Function](#content-cost-function)
         * [Style Cost Function](#style-cost-function)
         * [1D and 3D Generalizations](#1d-and-3d-generalizations)

