# Hyperparameter tuning, Batch Normalization and Programming Frameworks

### Tuning process

- We need some steps to tune our Hyperparameters to get the best out of them.
- So far the Hyperparameters importance are (As to Andrew Ng)
  1. Learning rate.
  2. Mini-batch size.
  3. No. of hidden units.
  4. Momentum beta.
  5. No. of layers.
  6. Use learning rate decay?
  7. Adam `beta1` & `beta2`
  8. regularization lambda
  9. Activation functions
- Its hard to decide which Hyperparameter is the more important in a problem. It depends much on your problem.
- One of the ideas to tune is to make a box with `N` Hyperparameter settings and then try the `N` settings on your problem.
- You can use Coarse to fine box and randomly initialize it the hyperparameters.
  - Then if you find some values that gives you a better values. Zoom into the box.
- This methods can be automated!

### Using an appropriate scale to pick hyperparameters

- If you have a specific range for a hyper parameter lets say from "a" to "b". Lets demonstrate the logarithmic scale, this will give you a good random points:
  - Calculate: `aLog = log(a)`                   `# Ex. a = 0.0001 then aLog = -4`
    - Calculate: `bLog = log(b)`                 `# Ex. b = 1  then bLog = 0`
  - Then: write this code:

    ```
    r = (aLog-bLog) * np.random.rand() + 	bLog
    # In our Ex the range would be from [-4, 0] because rand range [0,1)
    result = 10^r
    ```
It uniformly samples values from [a, b] as r.
- If we want to use the last method on exploring on the "momentum beta":
  - Beta best range is from 0.9 to 0.999
  - You should scale this to `1-Beta = 0.001 to 0.1` and the use `a = 0.001` and `b = 0.1`
  - And remember to subtract 1 from the resulted random value.

### Hyperparameters tuning in practice: Pandas vs. Caviar 

- If you don't have a much computational resources you will go thought "The baby sit model"
  - Run the model with different hyperparameters day by day.
  - Check at the end of each day if there are a progress or not.
  - You run one model at a time.
  - Called panda approach
- If you have computational resources, you can run some models in parallel and at the end of the day(s) you check the results.
  - Called Caviar model.

### Normalizing activations in a network

- In the current evolution of deep learning an algorithm called **Batch Normalization** is so important.
  - Made by Sergey Ioffe and Christian Szegedy.
- Batch Normalization speeds up learning.
- We discussed before that we can normalize input using the mean and variance method. This helped a lot in the shape of the cost function and reaching the minimum point in a more faster way!
- The question is *For any hidden layer can we normalize `A[l]` to train `W[l]`, `b[l]` faster?*. This is what batch normalization is about.
- Some papers normalize `Z[l]` and some normalize `A[l]`. Most of them uses `Z[l]` and recommended from Andrew Ng.
- Algorithm
  - Given `Z[l] = [z(1) z(2) .. z(m)]`   `#i = 1 to m (for one input)`
  - Compute `mean[i] = 1/m * sum(z[i])`
  - Compute `Variance[i] = 1/m * sum((z[i] - mean)^2)`
  - Then `Z_norm[i] = (z(i) - mean) / np.sqrt(Variance + epsilon)`
    - Forcing the outputs to a specific distribution.
  - Then `Z_dash[i] = alpha * Z_norm[i] + beta`
    - alpha and beta are learnable parameters.
    - Making the NN learn the distribution of the outputs.

### Fitting Batch Normalization into a neural network

- Using batch norm in 3 hidden layers NN:
  - ![](Images/bn.png)
- Our NN parameters will be:
  - `W[1]`, `b[1]`, `W[2]`, `b[2]`, `W[3]`, `b[3]`, `beta[1]`, `alpha[1]`, `beta[2]`, `alpha[2]`, `beta[3]`, `alpha[3]`
- If you are using a deep learning framework, You won't have to implement batch norm yourself.
  - Ex. in Tensorflow you can add this line: `tf.nn.batch-normalization()`
- If we are using batch norm the parameter `b[1]`, `b[2]`,.... Doesn't count because:
  - `Z[l] = W[l]A[l-1] + b[l]`
  - `Z_N[l] = alpha[l] * Z_norm[l] + beta[l]`
  - Taking the mean of a constant `b[l]` will eliminate the `b[l]`
- So if you are using batch normalization, you can remove b[l] or make it always zero.
- So the parameter will be Ws, betas, and alphas.
- Shapes:
  - `Z[l]`				`#(n[l], m)`
    - `alpha[l]`      	        `#(n[l], m)`
    - `beta[l]`                `#(n[l], m)`

### Why does Batch normalization work

- The first reason is the same reason as why we normalize X.
- The second reason is that batch normalization reduces the problem of input values changing.
- Batch norm does some regularization:
  - Each mini batch is scaled by the mean/variance computed of that mini batch.
  - -This adds some noise to the values `Z[l]` within that mini batch. Similar to dropout it adds some noise to each hidden layer activation
  - This has a slight regularization effect.
- To reduce this regularization effect you can make your mini batch bigger.
- If you need regularization you cant just rely on that slight regularization you'll need to add your regularization (L2 or dropout).

### Batch normalization at test time

- When we train a NN with Batch normalization, we compute the mean and the variance of the size of mini-batch.
- In testing we have to test one by one example. The mean and the variance of one example doesn't make sense!
- We have to compute an estimate value of mean and variance to use it in the testing time.
- We can use the weighted average across the mini batches.
- We will use the estimate values of the mean and variance to test.
- There are another method to estimate this value called "Running average"
- In practice don't worry as you will use a deep learning framework and it will contain some default of doing such a thing.

### Softmax Regression

- Every example we have used so far are talking about classification on only two classes.
- There are a generalization of logistic regression called Softmax regression that are more general.
- For example if we are classifying dogs, cat, and none of that
  - Dog `class = 1`
  - Cat `class = 2`
  - None `class = 0`
  - To represent a dog vector `y = [1 0 0]`
  - To represent a cat vector `y = [0 1 0]`
  - To represent a none vector `y = [0 0 1]`
- We will use these notations:
  - `C = no. Of classes`
  - Range of classes is `(0,...C-1)`
  - In output layer. `Ny = C`
- Each of the output layers will contain a probability if the class is true.
- In the last layer we will have to activate the Softmax activation function instead of the sigmoid activation.
- Softmax activation equations:

  ```
  t = e^(Z[L])        # shape(C, m)
  A[L] = e^(Z[L]) / sum(t, C)       # shape(C, m)
  ```

### Training a Softmax classifier

- There's an activation which is called hard max, which gets 1 for the maximum value and zeros for the others.
  - If you are using NumPy, its `np.max` over the vertical axis.
- The Softmax name came from Softening the values and not harding them like hard max.
- Softmax is a generalization of logistic regression with two or more classes.
- The loss function used with Softmax:

  ```
  L(y,y_dash) = -sum(y[i]*log(y_dash), C)
  ```

- The cost function used with Softmax:

  ```
  J(w[1], b[1], ....) = -1/m * (sum(L(y[i],y_dash[i]), m))
  ```

- Back propagation with Softmax:

  ```
  dZ[L] = Y_dash - Y
  ```

- The derivative of Softmax is:

  ```
  Y_dash( 1 - Y_dash)
  ```

- Example:
  - ![](Images/07-_softmax.png)

### Deep learning frameworks

- Its not practical to implement everything from scratch. Out last implementations was to know how NN works.
- There are many good deep learning frameworks.
- Deep learning is now in the phase of doing something with the frameworks and not from scratch to keep on going.
- Here are some of the leading deep learning frameworks:
  - Caffe/ Caffe2
  - CNTK
  - DL4j
  - Keras
  - Lasagne
  - mxnet
  - PaddlePaddle
  - TensorFlow
  - Theano
  - Torch/Pytorch
- These frameworks are getting better month by month. Comparison between them can be found [here](https://en.wikipedia.org/wiki/Comparison_of_deep_learning_software).
- How to choose deep learning framework:
  - Ease of programming (development and deployment)
  - Running speed
  - Truly open (Open source with good governance)
- Programing frameworks can not only shorten your coding time, but sometimes also perform optimizations that speed up your code.

### TensorFlow

- In this section we will learn the basic structure of TensorFlow.
- Lets see how implement a minimization function:
  - Function: `J(w) = w^2 - 10w + 25`
  - the result should be `w = 5` as the function is `(w-5)^2 = 0`
  - Code V1:

    ```python
    w = tf.Variable(0, dtype=tf.float32)                 # Creating a variable w
    cost = tf.add(tf.add(w**2, tf.multiply(-10.0, w)), 25.0)        # can be written as this [cost = w**2 - 10*w + 25]
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)
    session.run(w)    # Runs the definition of w, if you print this it will print zero
    session.run(train)

    print("W after one run: ", session.run(w))

    for i in range(1000):
    	session.run(train)

    print("W after 1000 run: ", session.run(w))
    ```

  - Code V2 (we feed the inputs to the algorithm through coefficient):

    ```python
    coefficient = np.array([[1.], [-10.], [25.]])

    x = tf.placeholder(tf.float32, [3, 1])
    w = tf.Variable(0, dtype=tf.float32)                 # Creating a variable w
    cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]

    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)
    session.run(w)    # Runs the definition of w, if you print this it will print zero
    session.run(train, feed_dict={x: coefficient})

    print("W after one run: ", session.run(w))

    for i in range(1000):
    	session.run(train, feed_dict={x: coefficient})

    print("W after 1000 run: ", session.run(w))
    ```

- In TensorFlow you implement the forward propagation and TensorFlow will do the back propagation because it knows how to do it.
- In TensorFlow a placeholder is a variable you can assign a value to later.
- If you are using a mini-batch training you should change the `feed_dict={x: coefficient}` to the current mini batch.
- Almost all TensorFlow  program uses this:

  ```python
  with tf.Session() as session:             # Because its better at clean up.
  	session.run(init)
  	session.run(w)
  ```

- In deep learning frameworks there are a lot of things that you can do with one line of code like changing the optimizer.
- Writing and running programs in TensorFlow has the following steps:
  1. Create Tensors (variables) that are not yet executed/evaluated.
  2. Write operations between those Tensors.
  3. Initialize your Tensors.
  4. Create a Session.
  5. Run the Session. This will run the operations you'd written above.

- Instead of needing to write code to compute the cost function we know, we can use this line in TensorFlow :

  `tf.nn.sigmoid_cross_entropy_with_logits(logits = ...,  labels = ...)`

- To initialize weights in NN using TensorFlow use:

  ```
  W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))

  b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
  ```

- For 3 layers NN, It is important to note that the forward propagation stops at `Z3`. The reason is that in TensorFlow the last linear layer output is given as input to the function computing the loss. Therefore, you don't need `A3`!
- To reset the graph
  - `tf.reset_default_graph()`


