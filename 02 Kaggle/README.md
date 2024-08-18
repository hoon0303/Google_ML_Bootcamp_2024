
# Binary Classification of Insurance Cross Selling(Playground Series - S4E7, Top 1%)

## Overview [[Kaggle]](https://www.kaggle.com/competitions/playground-series-s4e7/overview)

![Kaggle](Kaggle.PNG)

### Goal
> The objective of this competition is to predict which customers respond positively to an automobile insurance offer.



### Timeline
- Start Date - July 1, 2024
- Entry Deadline - Same as the Final Submission Deadline
- Team Merger Deadline - Same as the Final Submission Deadline
- Final Submission Deadline - July 31, 2024

### Evaluation
Submissions are evaluated using area under the ROC curve using the predicted probabilities and the ground truth targets.

### Submission File
For each `id` row in the test set, you must predict the probability of the target, `Response`. The file should contain a header and have the following format:

```bash
id, Response
11504798, 0.5
11504799, 0.5
11504800, 0.5
etc.
```

## Methodology

### 1. Data Preprocessing and Encoding Techniques

- **One-Hot Encoder**
- **Label Encoder**
- **Target Encoder**
- **MinMax Scaler**

### 2. Modeling Techniques

We experimented with a variety of machine learning models:

- **Multi-Layer Perceptron (MLP)**
- **CatBoost**
- **Artificial Neural Network (ANN)**
- **LightGBM**
- **Linear Regression**
- **XGBoost**
- **Random Forest**



### 3. Model Evaluation with K-Fold Cross-Validation
We evaluated the performance of each model using the **ROC AUC** score as the primary metric. To ensure that our models generalized well to unseen data, we utilized **K-fold cross-validation**. This approach effectively prevented overfitting and provided a robust evaluation of the model's performance. After comparing various models, **CatBoost** and **XGBoost** showed the best performance. These models were further fine-tuned through hyperparameter optimization and used as the base for more advanced techniques.

### 4. Stacking and Ensemble Techniques
To further enhance the predictive performance, we employed **stacking** as an ensemble method. By combining the predictions of multiple models (including CatBoost, LightGBM, XGBoost, etc.) into a meta-model, we were able to leverage the strengths of each model, resulting in a more robust and accurate final prediction.

## Reference
- Walter Reade, Ashley Chow. (2024). Binary Classification of Insurance Cross Selling . Kaggle. https://kaggle.com/competitions/playground-series-s4e7
