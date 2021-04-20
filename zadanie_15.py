import numpy as np
import matplotlib.pyplot as plt

def Exc_no_15(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] < 0:
            y[i] = np.sin(x[i])
        else:
            y[i] = np.sqrt(x[i])
    return y
