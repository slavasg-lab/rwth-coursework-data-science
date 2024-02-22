import numpy as np
import math as math

class DataTransformer:
    def __init__(self, activate):
        self.activate = activate

    def transform(self, inData):
        if not self.activate:
            return inData
        else:
            # 2D array to 3D array by squaring 2D points and taking result as 3rd value
            sq_sum = np.square(inData.iloc[:, 0]) + np.square(inData.iloc[:, 1])
            outData= np.c_[inData, sq_sum]
        return outData