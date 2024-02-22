import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class Plot:
    def __init__(self):
        # Plot - Class provides an empty canvas
        self.fig = plt.figure()
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        # self.ax = plt.gca()
        plt.xlabel("x_1", fontsize="x-large")
        plt.ylabel("x_2", fontsize="x-large")

    # ------------------------------------------------
    # plotCorrectLabels plots 2D scatter plot of
    # correct class lables of input data.
    # ------------------------------------------------
    def plotCorrectLabels(self, inputs, labels):
        X = inputs[:, 0]  # x-values
        Y = inputs[:, 1]  # y-values
        scatter = self.fig.add_subplot(1, 1, 1)
        scatter.scatter(X, Y, c=labels, cmap='bwr')
        return scatter  # returns true labels

    # ------------------------------------------------
    # plot2DClassifier plots the perceptron decision boundary
    # ------------------------------------------------

    def plot2DClassifier(self, inputs, weights):
        x = inputs[:, 0]  # x-values

        # Avoiding numerical errors
        weights[weights == 0] += np.finfo(np.float32).eps

        # Converting weights and bias to decision boundary
        slope = - weights[1] / weights[2]
        intercept = - weights[0] / weights[2]
        xx = np.linspace(min(x), max(x))
        yy = slope * xx + intercept
        classifier = self.fig.add_subplot(1, 1, 1)
        classifier.plot(xx, yy, 'k-')
        plt.grid(True, which='both')
        return classifier
