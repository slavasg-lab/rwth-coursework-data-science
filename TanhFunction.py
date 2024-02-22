import numpy as np

class TanhFunction:
    def activation(self, x):
        """TODO: Implementieren Sie die tanh Funktion"""
        return np.tanh(x)

    def gradient(self, x):
        """TODO: Implementieren Sie den Gradienten der tanh Funktion"""
        gradient = 1 - np.square(np.tanh(x))
        return gradient