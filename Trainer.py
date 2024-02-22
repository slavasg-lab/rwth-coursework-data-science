"""
Trainer is the main entry point to the training of the perceptron, calling
the data loader, the perceptron and the plotting utilities.
"""
import pandas as pd
import numpy as np
from Perceptron import Perceptron
from DataTransformer import DataTransformer
from SignFunction import SignFunction
from TanhFunction import TanhFunction

# Reading the input-target pairs from the data file
data_file = pd.read_csv("data.csv", header=None)

# extract inputs and labels from read data
inputs = data_file.iloc[:, 0:2]  # observation
labels = data_file.iloc[:, 2]

# Transformation
transformData = False  # whether to transform the data before training the perceptron or not
bf = DataTransformer(transformData)
inputs = bf.transform(inputs)

learning_rate = 0.01
tanh_activation = TanhFunction()
sign_activation = SignFunction()

perceptron = Perceptron(np.shape(inputs)[1], learning_rate=learning_rate, activation=tanh_activation)
perceptron.train(inputs=inputs, labels=labels, epochs=100)
