# Structuring Machine Learning Projects

This is the third course of the deep learning specialization at [Coursera](https://www.coursera.org/specializations/deep-learning) which is moderated by [DeepLearning.ai](http://deeplearning.ai/). The course is taught by Andrew Ng.

## Course summary

Here are the course summary as its given on the course [link](https://www.coursera.org/learn/machine-learning-projects):

> You will learn how to build a successful machine learning project. If you aspire to be a technical leader in AI, and know how to set direction for your team's work, this course will show you how.
>
> Much of this content has never been taught elsewhere, and is drawn from my experience building and shipping many deep learning products. This course also has two "flight simulators" that let you practice decision-making as a machine learning project leader. This provides "industry experience" that you might otherwise get only after years of ML work experience.
>
> After 2 weeks, you will: 
> - Understand how to diagnose errors in a machine learning system, and 
> - Be able to prioritize the most promising directions for reducing error
> - Understand complex ML settings, such as mismatched training/test sets, and comparing to and/or surpassing human-level performance
> - Know how to apply end-to-end learning, transfer learning, and multi-task learning
>
> I've seen teams waste months or years through not understanding the principles taught in this course. I hope this two week course will save you months of time.
>
> This is a standalone course, and you can take this so long as you have basic machine learning knowledge. This is the third course in the Deep Learning Specialization.

## Table of contents

* [ML Strategy 1](#ml-strategy-1)
   * [Why ML Strategy](#why-ml-strategy)
   * [Orthogonalization](#orthogonalization)
   * [Single number evaluation metric](#single-number-evaluation-metric)
   * [Satisfying and Optimizing metric](#satisfying-and-optimizing-metric)
   * [Train/Dev/Test distributions](#traindevtest-distributions)
   * [Size of the Dev and Test sets](#size-of-the-dev-and-test-sets)
   * [When to change Dev/Test sets and metrics](#when-to-change-devtest-sets-and-metrics)
   * [Why human-level performance?](#why-human-level-performance)
   * [Avoidable bias](#avoidable-bias)
   * [Understanding human-level performance](#understanding-human-level-performance)
   * [Surpassing human-level performance](#surpassing-human-level-performance)
   * [Improving your model performance](#improving-your-model-performance)
* [ML Strategy 2](#ml-strategy-2)
   * [Carrying out error analysis](#carrying-out-error-analysis)
   * [Cleaning up incorrectly labeled data](#cleaning-up-incorrectly-labeled-data)
   * [Build your first system quickly, then iterate](#build-your-first-system-quickly-then-iterate)
   * [Training and testing on different distributions](#training-and-testing-on-different-distributions)
   * [Bias and Variance with mismatched data distributions](#bias-and-variance-with-mismatched-data-distributions)
   * [Addressing data mismatch](#addressing-data-mismatch)
   * [Transfer learning](#transfer-learning)
   * [Multi-task learning](#multi-task-learning)
   * [What is end-to-end deep learning?](#what-is-end-to-end-deep-learning)
   * [Whether to use end-to-end deep learning](#whether-to-use-end-to-end-deep-learning)

