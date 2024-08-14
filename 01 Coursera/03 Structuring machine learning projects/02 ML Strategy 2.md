# ML Strategy 2

### Carrying out error analysis

- Error analysis is to analysis why the accuracy of the system is like that. Example:
  - In the cat classification example, if you have 10% error on your Dev set and you want to solve the error.
  - If you discovered that some of the mislabeled data are dog pictures that looks like cats, should you try to make your cat classifier do better on dogs? this could take some weeks.
  - Error analysis approach (To take a decision):
    - Get 100 mislabeled Dev set examples at random.
    - Count up how many are dogs.
    - if there are 5/100 is dogs then it doesn't count to train your classifier to dogs.
    - if there are 50/100 is dogs then you should work in that.
- Based on the last example, error analysis helps you to analyze the error before taking an action that could take lot of time with no need.
- You can evaluate multiple ideas -Error analysis ideas- in parallel and choose the best idea. create an excel shape to do that and decide Ex:

  | Image        | Dog    | Great Cats | blurry  | Comments |
  | ------------ | ------ | ---------- | ------- | -------- |
  | 1            | ✓      |            |         |          |
  | 2            | ✓      |            | ✓       |          |
  | 3            |        |            |         |          |
  | 4            |        | ✓          |         |          |
  | ....         |        |            |         |          |
  | **% totals** | **8%** | **43%**    | **61%** |          |

- In the last example you will decide to work on great cats or blurry images to improve your performance.

### Cleaning up incorrectly labeled data

- Labeled data is incorrect when y of x is incorrect.
- If the incorrect labeled data is in the training set, Deep learning are quite robust to random error (Not systematic error). But its OK to go and fix these labels if you can.
- If you want to check for mislabeled data in Dev/test set, you should also try error analysis with mislabeled column. Ex:

  | Image        | Dog    | Great Cats | blurry  | Mislabeled | Comments |
  | ------------ | ------ | ---------- | ------- | ---------- | -------- |
  | 1            | ✓      |            |         |            |          |
  | 2            | ✓      |            | ✓       |            |          |
  | 3            |        |            |         |            |          |
  | 4            |        | ✓          |         |            |          |
  | ....         |        |            |         |            |          |
  | **% totals** | **8%** | **43%**    | **61%** | **6%**     |          |

  - Then:
    - If Overall Dev set error: 		10%
      - Then Errors due incorrect data: 0.6%
        - Then Errors due other causes:9.4%
    - Then you should focus on the 9.4% error rather than the incorrect data.
- Consider these while correcting the Dev/test mislabeled:
  - Apply same process to your Dev and test sets to make sure they continue to come from the same distribution.
  - Consider examining examples your algorithm got right as well as ones it got wrong. (Not always done if you reached a good accuracy)
  - Train and (Dev/Test) data may now come from slightly different distributions

### Build your first system quickly, then iterate

- The steps you take to make your deep learning project:
  - Setup Dev/test set and metric
  - Build initial system quickly
    - Using the training data.
  - Use Bias/Variance analysis & Error analysis to prioritize next steps.

### Training and testing on different distributions

- A lot of teams are working with deep learning applications that has training sets that are different from the Dev/test sets due to the hanger of deep learning to data.
- There are some strategies to follow up when training set distribution differs from Dev/test sets distribution.
  - Option one (Not recommended): shuffle all the data together and extract randomly training and Dev/test sets.
    - Advantages:   All the sets now are from the same distribution.
    - Disadvantages: The other distribution that was in the Dev/test sets will occur less in the new Dev/test sets and that might not what you want to achieve.
  - Option two: Take some of the Dev/test set examples and put them with the training distribution.
    - Advantages: The distribution you care about is your target now.
    - Disadvantage: the distributions are different. but you will get a better performance over a long time.

### Bias and Variance with mismatched data distributions

- Bias and Variance analysis changes when training and Dev/test set is from different distribution.
- Example: Assume the cat classification example. Suppose you've worked in the example and reached this
  - Human error:               0%
  - Training error:              1%
  - Dev error:                      10%
- In the last example you'll think that this is a variance problem, but because the distributions aren't the same you cant judge this.
- Imagine if we created a new set called training-Dev set as a random subset of the training distribution. and we run error analysis and it came as follows:
  - Human error:               0%
  - Training error:              1%
  - TrainingDev error:       8%
  - Dev error:                      10%
- Now you are sure this is a variance error.
- Suppose you have a different situation:
  - Human error:               0%
  - Training error:              1%
  - TrainingDev error:       1.5%
  - Dev error:                      10%
- In this case you have something called *Data mismatch* problem.
- To conclude, first you'll have a new set called training-Dev set which has the same distribution as training set. Then follow this:
  1. human level error (Proxy for Bayes error)
     - Calculate `training error - human level error`
     - If difference is bigger then its **Avoidable bias** then you should use a strategy for **bias**.
  2. Training error
     - Calculate `Training-Dev error - training error`
     - If difference is bigger then its **Variance** then you should use a strategy for **Variance**.
  3. Training-Dev error
     - Calculate `dev error - training-dev error`
     - If difference is bigger then its **Data mismatch** then you should use a strategy for **Data mismatch**.
  4. Dev error
     - Calculate `test error - dev error`
     - Is the degree of overfitting to Dev set
  5. Test error
- Unfortunately there aren't much systematic ways to deal with Data mismatch but the next section will try to give us some insights.

### Addressing data mismatch

- Carry out manual error analysis to try to understand difference between training and Dev/test sets.
- Make training data more similar; or collect more data similar to Dev/test sets.
  - There are something called **Artificial data synthesis** that can help you Make more training data.
    - Combine some of your training data with something that can convert it to the Dev/test set distribution.
      - Ex. Generate cars using 3D in a car classification example.
    - Be careful with "Artificial data synthesis" because your NN might overfit these generated data.

### Transfer learning

- Apply the knowledge you took in a task and apply it in another task.
- For example You have trained a cat classifier with a lot of data, you can use all the learning data or part of it to solve x-ray classification problem.
- To do transfer learning, delete the weights of the last layer of the NN and keep all the other weights as a fixed weights. Initialize the new weights and feed the new data to the NN and learn the new weights. Thats if you have a small data set, but if you have enough data you can retrain all the weights again this is called **fine tuning**.
- You can create a several new layers not just one layer to original NN.
- When transfer learning make sense:
  - When you have a lot of data for the problem you are transferring from and relatively less data for the problem your transferring to.
  - Task A and B has the same input X.   (Same type as input ex. image, audio)
  - Low level features from A could be helpful for learning B

### Multi-task learning

- One NN do some tasks in the same time, and tasks can help each others.
- Example:
  - You want to build an object recognition system that detects cars, stop signs, and traffic lights. (Image has a multiple labels.)
  - Then Y shape will be `(3,m)` because we have 3 classification and each one is a binary one.
  - Then `Loss = (1/m) sum(sum(L(Y_dash[i], Y[i]),3) ,m)`
- In the last example you could have train 3 neural network to get the same results, but if you suspect that the earlier layers has the same features then this will be faster.
- This will also work if y isn't complete for some labels. For example:

  ```
  Y = [1	?	1	..]
      [0	0	1	..]
      [?	1	?	..]
  ```

  - And in this case it will do good with the missing data. but the loss function will be different:
    - `Loss = (1/m) sum(sum(L(Y_dash[i], Y[i]),for all i which Y[i] != ?) ,m)`


- When Multi-task learning make sense:
  - Training on a set of tasks that could benefit from having shared lower-level features.
  - Usually amount of data you have for each task is quite similar.
  - Can train a big enough network to do well on all the tasks.
- If you have a big enough NN, the performance of the Multi-task learning compared to splitting the tasks is better.
- Today Transfer learning is used more than Multi-task learning.

### What is end-to-end deep learning?

- Some systems has multiple stages to implement. An end to end deep learning implements all these stages with a single NN.
- Example:
  - Suppose you have a speech recognition system:

    ```
    Audio ---> Features --> Phonemes --> Words --> Transcript			# System
    Audio ---------------------------------------> Transcript			# End to end
    ```

  - End to end deep learning gives data more freedom, it might not use phonemes when training!
- To build and end to end deep learning system that works well, we need a big dataset. If you have a small dataset the ordinary implementation of each stage is just fine.

- Another example:
  - Suppose you want to build a face recognition system:

    ```
    Image->Image adjustments->Face detection->Face recognition->Matching	# System.
    Image ----------------------------------->Face recognition->Matching  # End to end
    Image->Image adjustments->Face detection------------------->Matching  # Best imp for now
    ```

  - Best in practice now is the third approach.
  - In the third implementation its a two steps approach where part is manually implemented and the other is using deep learning.
  - Its working well because its harder to get a lot of pictures with people in front of the camera than getting faces of people and compare them.
  - In the third implementation the NN takes two faces as an input and outputs if the two faces are the same or not.
- Another example:
  - Suppose you want to build a machine translation system:

    ```
    English --> Text analysis --> ......................... --> Fresh		# System.
    English --------------------------------------------------> Fresh		# End to end
    ```

  - Here end to end deep leaning system works well because we have enough data to build it.

### Whether to use end-to-end deep learning

- Here are some guidelines on Whether to use end-to-end deep learning.
- Pros of end to end deep learning:
  - Let the data speak.
  - Less hand designing of components needed.
- Cons of end to end deep learning:
  - May need large amount of data.
  - Excludes potentially useful hand design components. (It helps more on small dataset)
- Applying end to end deep learning:
  - Do you have sufficient data to learn a function of the ***complexity*** needed to map x to y?
