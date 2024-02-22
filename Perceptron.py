import numpy as np
from Plot import Plot
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, inputDim, activation, learning_rate=0.01):
        self.activation = activation
        self.inputDim = inputDim  # dimension of the inputs (2 in the case of 2D points)
        self.learning_rate = learning_rate
        self.weights = np.zeros(self.inputDim + 1)  # add one for bias, np.random.rand(inputDim + 1)
        self.initializeWeights(self.inputDim)

    # ------------------------------------------------
    # Method to initialize the weights and the bias
    # ------------------------------------------------
    def initializeWeights(self, inputDim):
        """TODO: Weisen Sie den Gewichten und dem Bias des Perzeptron zufaellige Werte zwischen -1 und 1 zu

        INPUT: inputDim: Dimension der Input"""
        self.weights = np.random.rand(inputDim + 1) * 2 - 1

        # -----------------------------------------------

    # Method to compute Dotproduct + Bias s
    # ------------------------------------------------
    def dotProductWeigthsPlusBias(self, input_data):
        z = self.weights[0]  # bias
        z += self.weights[1:] @ input_data
        return z

    # ------------------------------------------------
    # Method to feedforward an input sample
    # through the perceptron
    # ------------------------------------------------
    def feedforward(self, input_data):
        """TODO: Berechnen Sie die gewichtete Summe der Eingabewerte

        INPUT:
            input_data: ein Datenpunkt
            z: Skalarprodukt des Gewichtsvektors mit den Inputs (plus Bias)
        OUTPUT:
            Vorhersage der Aktivierungsfunktion"""
        z = self.dotProductWeigthsPlusBias(input_data)
        return self.activation.activation(z)

    # ------------------------------------------------
    # Method to compute (squared) error of
    # one prediction target pair
    # ------------------------------------------------
    def errorFunction(self, prediction, target):
        """TODO: Berechnen Sie den quadratischen Fehler

        INPUT:
               prediction: vorhergesagte Klassenzugehoerigkeiten
               target: wahre Klassifikationslabel
        OUTPUT:
               Quadratischer Fehler"""
        error = 0.5 * np.square(target - prediction)
        return error

    # ------------------------------------------------
    # Method to plot progress
    # ------------------------------------------------
    def plotProgress(self, inputs, labels):  # progress_figure
        figure = Plot()
        if (self.inputDim == 2):
            figure.plotCorrectLabels(inputs, labels)
            figure.plot2DClassifier(inputs, self.weights)
            plt.show()
        else:  # non-lin cases (circle-data, dsme-data)
            # calculate predicted labels
            pred_labels = np.zeros(np.shape(inputs)[0])
            for j in range(np.shape(inputs)[0]):
                x = inputs[j, :]  # jth example inputs.iloc[:, 0]
                pred_labels[j] = self.feedforward(x)
            figure.plotCorrectLabels(inputs, pred_labels)
            plt.show()

    # ------------------------------------------------
    # Method to train perceptron
    # ------------------------------------------------

    def train(self, inputs, labels, epochs=40):
        inputLen = np.shape(inputs)[0]  # number of input-target pairs for training
        # Train the perceptron for the number of epochs
        for epochs in range(epochs):

            # shuffle training data
            tmp = shuffle(np.column_stack((inputs, labels)), random_state=0)

            inputs = tmp[:, 0:self.inputDim]
            labels = tmp[:, -1]

            # initialization of loss
            loss = 0  # accumulated error for all samples in one epoch

            # Loop over all samples
            for j in range(inputLen):

                # TODO: HIER CODE ERGAENZEN

                # 1. Berechnung der Outputs mit Hilfe der feedforward Methode
                x = inputs[j, :]
                prediction = self.feedforward(x)

                # 2. Berechnung der Summer der Fehler fuer jeden Datenpunkt
                loss += self.errorFunction(prediction, labels[j])

                # 3. Berechnung der Lernregel / des Gradienten
                z = self.dotProductWeigthsPlusBias(x)
                gradient = self.activation.gradient(z)

                # 4. Update der Gewichte mit Hilfe der Updateregel
                self.weights[0] += self.learning_rate * (labels[j] - prediction) * gradient
                for c, e in enumerate(x):
                    self.weights[c+1] += self.learning_rate * (labels[j] - prediction) * gradient * e

            # We are plotting the current classifier for each epoch
            self.plotProgress(inputs, labels)  # labels = true labels

            print("Epoch: ", epochs, " Loss: ", loss)

            # TODO: Beendung des Trainings, wenn der Gesamtfehler gleich 0 ist

            if loss == 0:
                break
