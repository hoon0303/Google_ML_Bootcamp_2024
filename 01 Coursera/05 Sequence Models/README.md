# Sequence Models

This is the fifth and final course of the deep learning specialization at [Coursera](https://www.coursera.org/specializations/deep-learning) which is moderated by [deeplearning.ai](http://deeplearning.ai/). The course is taught by Andrew Ng.

## Course summary
Here are the course summary as its given on the course [link](https://www.coursera.org/learn/nlp-sequence-models):

> This course will teach you how to build models for natural language, audio, and other sequence data. Thanks to deep learning, sequence algorithms are working far better than just two years ago, and this is enabling numerous exciting applications in speech recognition, music synthesis, chatbots, machine translation, natural language understanding, and many others. 
>
> You will:
> - Understand how to build and train Recurrent Neural Networks (RNNs), and commonly-used variants such as GRUs and LSTMs.
> - Be able to apply sequence models to natural language problems, including text synthesis. 
> - Be able to apply sequence models to audio applications, including speech recognition and music synthesis.
>
> This is the fifth and final course of the Deep Learning Specialization.

## Table of contents
* [Sequence Models](#sequence-models)
   * [Table of contents](#table-of-contents)
   * [Course summary](#course-summary)
   * [Recurrent Neural Networks](#recurrent-neural-networks)
      * [Why sequence models](#why-sequence-models)
      * [Notation](#notation)
      * [Recurrent Neural Network Model](#recurrent-neural-network-model)
      * [Backpropagation through time](#backpropagation-through-time)
      * [Different types of RNNs](#different-types-of-rnns)
      * [Language model and sequence generation](#language-model-and-sequence-generation)
      * [Sampling novel sequences](#sampling-novel-sequences)
      * [Vanishing gradients with RNNs](#vanishing-gradients-with-rnns)
      * [Gated Recurrent Unit (GRU)](#gated-recurrent-unit-gru)
      * [Long Short Term Memory (LSTM)](#long-short-term-memory-lstm)
      * [Bidirectional RNN](#bidirectional-rnn)
      * [Deep RNNs](#deep-rnns)
      * [Back propagation with RNNs](#back-propagation-with-rnns)
   * [Natural Language Processing &amp; Word Embeddings](#natural-language-processing--word-embeddings)
      * [Introduction to Word Embeddings](#introduction-to-word-embeddings)
         * [Word Representation](#word-representation)
         * [Using word embeddings](#using-word-embeddings)
         * [Properties of word embeddings](#properties-of-word-embeddings)
         * [Embedding matrix](#embedding-matrix)
      * [Learning Word Embeddings: Word2vec &amp; GloVe](#learning-word-embeddings-word2vec--glove)
         * [Learning word embeddings](#learning-word-embeddings)
         * [Word2Vec](#word2vec)
         * [Negative Sampling](#negative-sampling)
         * [GloVe word vectors](#glove-word-vectors)
      * [Applications using Word Embeddings](#applications-using-word-embeddings)
         * [Sentiment Classification](#sentiment-classification)
         * [Debiasing word embeddings](#debiasing-word-embeddings)
   * [Sequence models &amp; Attention mechanism](#sequence-models--attention-mechanism)
      * [Various sequence to sequence architectures](#various-sequence-to-sequence-architectures)
         * [Basic Models](#basic-models)
         * [Picking the most likely sentence](#picking-the-most-likely-sentence)
         * [Beam Search](#beam-search)
         * [Refinements to Beam Search](#refinements-to-beam-search)
         * [Error analysis in beam search](#error-analysis-in-beam-search)
         * [BLEU Score](#bleu-score)
         * [Attention Model Intuition](#attention-model-intuition)
         * [Attention Model](#attention-model)
      * [Speech recognition - Audio data](#speech-recognition---audio-data)
         * [Speech recognition](#speech-recognition)
         * [Trigger Word Detection](#trigger-word-detection)
  