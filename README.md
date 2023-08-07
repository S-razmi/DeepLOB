# Implementing DeepLOB on BTC perpetual LOB data:
This project focuses on implementing the ["DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"](https://arxiv.org/abs/1808.03668) paper by Zhang et al. on BTCUSDT Perpetual data. The model is trained on the one-second timeframe of LOB data, and the results are reviewed. A Temporal Convolutional Neural Network (TCN) is also designed and trained on the same data, and the results are compared with the original model. The code is implemented in TensorFlow using Python.

## Data Description
The analysis was encompassed by a two-week span of Limit Order Book (LOB) samples, each taken at a 25-millisecond interval, incorporating 20 columns for both bid and ask prices. However, due to its huge size, the data had to be resampled to 1-second intervals. The Dataset is available on [Kaggle](https://www.kaggle.com/datasets/siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data)
## Normalization
The same approach outlined in the original paper was followed for normalization purposes. Specifically, a sliding window with a length of one day was utilized to calculate the mean and standard deviation of past data. This enabled the current row to be scaled based on those values.
## Labeling
An averaging method is employed to generate mid-price first levels of bid and ask. The labeling process is carried out using the first method stated in the paper. A threshold alpha value of 0.00001 is also utilized, with a horizon of 10.
## Models
This project employed two models for training: the original model from the referenced paper and a novel model based on the Temporal Convolutional Neural Network (TCN) architecture. Both models underwent a training process of 10 epochs, with the same Dataset being utilized to ensure a fair comparison.

The original model from the paper was adopted as the starting point, providing a reliable baseline for the research. Additionally, a TCN-based model was designed to explore the potential of enhancing predictive capabilities in the task. TCNs have demonstrated promise in various time-series tasks due to their ability to efficiently model long-range dependencies in sequential data, making them a compelling choice for experimentation in this project.

## Results


For objective evaluation, a separate out-of-sample dataset, not used during training, was employed to validate the generalization abilities of each model and prevent potential biases from the training data. The results obtained from testing both models on this out-of-sample data were compared using various performance metrics, including accuracy, precision, recall, and F1-score, providing a comprehensive understanding of the strengths and weaknesses of each approach.
