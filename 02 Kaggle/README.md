
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
