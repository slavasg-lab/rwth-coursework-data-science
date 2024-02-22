class SignFunction:
    def activation(self, x):
        """OUTPUT: 1 oder -1
        TODO: Implementieren Sie die Sign Funktion"""
        return 1 if x >= 0 else -1

    def gradient(self, x):
        """TODO: Implementieren Sie den Gradienten der Sign Funktion"""
        return 1
