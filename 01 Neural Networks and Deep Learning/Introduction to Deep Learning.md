# Introduction to Deep Learning

### What is a (Neural Network) NN?

- Single neuron == linear regression
- Simple NN graph:
  - ![](Images/Others/01.jpg)
  - Image taken from [tutorialspoint.com](tutorialspoint.com)
- RELU stands for rectified linear unit is the most popular activation function right now that makes deep NNs train faster now.
- Hidden layers predicts connection between inputs automatically, thats what deep learning is good at.
- Deep NN consists of more hidden layers (Deeper layers)
  - ![](Images/Others/02.png)
  - Image taken from [opennn.net](opennn.net)
- Each Input will be connected to the hidden layer and the NN will decide the connections.
- Supervised learning means we have the (X,Y) and we need to get the function that maps X to Y.

### Supervised learning with neural networks

- Different types of neural networks for supervised learning which includes:
  - CNN or convolutional neural networks (Useful in computer vision)
  - RNN or Recurrent neural networks (Useful in Speech recognition or NLP)
  - Standard NN (Useful for Structured data)
  - Hybrid/custom NN or a Collection of NNs types
- Structured data is like the databases and tables.
- Unstructured data is like images, video, audio, and text.
- Structured data gives more money because companies relies on prediction on its big data.

### Why is deep learning taking off?

- Deep learning is taking off for 3 reasons:
  1. Data:
     - Using this image we can conclude:
       - ![](Images/11.png)
     - For small data NN can perform as Linear regression or SVM (Support vector machine)
     - For big data a small NN is better that SVM
     - For big data a big NN is better that a medium NN is better that small NN.
     - Hopefully we have a lot of data because the world is using the computer a little bit more
       - Mobiles
       - IOT (Internet of things)
  2. Computation:
     - GPUs.
     - Powerful CPUs.
     - Distributed computing.
     - ASICs
  3. Algorithm:
     1. Creative algorithms has appeared that changed the way NN works.
        - For example using RELU function is so much better than using SIGMOID function in training a NN because it helps with the vanishing gradient problem.