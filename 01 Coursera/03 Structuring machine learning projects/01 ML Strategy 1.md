# ML Strategy 1

### Why ML Strategy

- You have a lot of ideas to improve the accuracy of your deep learning system:
  - Collect more data.
  - Collect more diverse training set.
  - Train gradient decent longer.
  - Try bigger network.
  - Try smaller network.
  - Try dropout
  - Add L2 regularization
  - Try different optimization algorithm "ex. Adam"
  - Activation functions.
- This course will give you some strategies to help you analyze your problem to go in a direction that will get you a better results.

### Orthogonalization

- Some deep learning developers knows exactly what hyperparameter to tune to achieve a specific task. This is called Orthogonalization.
- In Orthogonalization you have some controls, but each control does a specific task and doesn't effect other controls.
- Chain of assumptions in machine learning:
  1. You'll have to fit training set well on cost function. (Near human level performance if possible)
     - If its not achieved you could try: bigger network - other optimization algorithm...
  2. Fit Dev set well on cost function.
     - If its not achieved you could try: regularization - Bigger training set ...
  3. Fit test set well on cost function.
     - If its not achieved you could try: Bigger Dev. set ...
  4. Performs well in real world.
     - If its not achieved you could try: change dev. set - change cost function..

### Single number evaluation metric

- Its better and faster to set a Single number evaluation metric to your project before you start it.
- Difference between precision and recall (In cat classification example):
  - Suppose we run the classifier on 10 images which are 5 cats and 5 non-cats. The classifier identifies that there are 4 cats. but he identified 1 wrong cat.

  - Confusion matrix:

    - |             | Cat  | Non-Cat |
      | ----------- | ---- | ------- |
      | **Cat**     | 3    | 2       |
      | **Non-Cat** | 1    | 4       |

  - **Precision**: percentage of true cats in the recognized result. per = 3/4

  - **Recall**: percentage of true recognition in the whole dataset. rec = 3/5

  - **Accuracy**= 3/10
- Using a precision/recall for evaluation is good in a lot of cases they doesn't tell you which is better. Ex:

  | Classifier | Precision | Recall |
  | ---------- | --------- | ------ |
  | A          | 95%       | 90%    |
  | B          | 98%       | 85%    |

- A better thing is to merge precision and Recall together. There a something called `F1` score
  - You can think of `F1` score as average of Precision and Recall
    `F1 = 2/ ((1/Per) + (1/Rec))`
- IF you have a lot of value as your metric  you should take the average.

### Satisfying and Optimizing metric

- Its hard sometimes to get a single number evaluation metric. Ex:

  | Classifier | F1   | Running time |
  | ---------- | ---- | ------------ |
  | A          | 90%  | 80 ms        |
  | B          | 92%  | 95 ms        |
  | C          | 92%  | 1,500 ms     |

- In this case we can solve that by Satisfying and Optimizing metric. Ex:

  ```
  Maximize 		F1							# Optimizing metric
  subject to 		Running time < 100ms		# Satisficing metric
  ```

- So as a general rule:

  ```
  Maximize 		1					#Optimizing metric (One optimizing metric)
  subject to 		N-1					#Satisficing metric (N-1 Satisficing metric)
  ```

### Train/Dev/Test distributions

- Dev/Test set has to come from the same distribution.
- Choose Dev/Test sets to reflect data you expect to get in the future and consider important to do well on.

### Size of the Dev and Test sets

- Old way of splitting was 70% training, 30% test.
- Old way of splitting was 60% training, 20% Dev, 20% test.
- The old way was valid for ranges 1000 --> 100000
- In the modern deep learning you have if you have a million or more
  - 98% Training, 1% Dev, 1% Test

### When to change Dev/Test sets and metrics

- Lets take an example. In a cat classification example we have these metric results:

  | Metric      | Classification error                     |
  | ----------- | ---------------------------------------- |
  | Algorithm A | 3% error (But a lot of porn images is treated as cat images here) |
  | Algorithm B | 5% error                                 |
  - In the last example if we choose the best algorithm by metric it would be "A", but if the users decide it will be "B"

  - Thus here we want to change out metric.
  - `OldMetric = (1/m) * sum(y_pred[i] != y[i] ,m)`
    - Where m is the number of Dev set items.
  - `NewMetric = (1/sum(w[i])) * sum( w[i] * (y_pred[i] != y[i]) ,m)`
    - where:
       - `w[i] = 1                   if x[i] is not porn`
       - `w[i] = 10                 if x[i] is porn`

- Conclusion: If doing well on your metric + Dev/test set doesn't correspond to doing well in your application, change your metric and/or Dev/test set.

### Why human-level performance?

- We compare to human-level performance because a lot of deep learning algorithms in the recent days are a lot better than human level.
- After an algorithm reaches the human level performance it doesn't get better much.
  - ![01- Why human-level performance](Images/01-_Why_human-level_performance.png)
- You won't surpass an error that's called "Bayes optimal error"
- There aren't much error range between human-level error and Bayes optimal error.
- Humans are quite good at lot of tasks. So as long as Machine learning is worse than humans, you can:
  - Get labeled data from humans.
  - Gain insight from manual error analysis. (Why did a person get it right?)
  - Better analysis of bias/variance

### Avoidable bias

- Suppose that the cat classification algorithm gives these percentages:

  | Humans             | 1%   | 7.5% |
  | ------------------ | ---- | ---- |
  | **Training error** | 8%   | 8%   |
  | **Dev Error**      | 10%  | 10%  |

  - In the left example, if the human level error is 1% then we have to focus on the **bias**.
  - In the right example, if the human level error is 7.5% then we have to focus on the **variance**.
  - In the latest examples we have used the human level as a proxy form Bayes optimal error because humans vision is too good.

### Understanding human-level performance

- When choosing human-level performance, it has to be choose in the terms of what you want to achieve with the system.
- You might have multiple human-level performance based on the human experience. Then the system you are trying to build will choose from these human levels as set it as proxy for Bayes error.
- Improving deep learning algorithms is harder once you reach a human level performance.
- Summary of bias/variance with human-level performance:
  1. human level error (Proxy for Bayes error)
     - Calculate `training error - human level error`
     - If difference is bigger then its **Avoidable bias** then you should use a strategy for **bias**.
  2. Training error
     - Calculate `dev error - training error`
     - If difference is bigger then its **Variance** then you should use a strategy for **Variance**.
  3. Dev error
- In a lot of problems Bayes error isn't zero that's why we need human level performance comparing.

### Surpassing human-level performance

- In some problems, deep learning has surpassed human level performance. Like:
  - Online advertising.
  - Product recommendation.
  - Loan approval.
- The last examples are non natural perception task. Humans are far better in natural perception task like computer vision and speech recognition.
- Its harder for machines to surpass human level in natural perception task.

### Improving your model performance

- To improve your deep learning supervised system follow these guideline:
  1. Look at the difference between human level error and the training error.  ***Avoidable bias***
  2. Look at the difference between the training error and the Test/Dev set. ***Variance***
  3. If number 1 difference is large you have these options:
     - Train bigger model.
     - Train longer/better optimization algorithm (Adam).
     - NN architecture/hyperparameters search.
     - Bigger training data.
  4. If number 2 difference is large you have these options:
     - Get more training data.
     - Regularization.
     - NN architecture/hyperparameters search.
