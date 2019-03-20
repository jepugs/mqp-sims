import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plotAccuracyToParam(accuracy, param_data, param):
    plt.plot(param_data, accuracy, 'ro-')
    plt.ylabel('Prediction accuracy')
    plt.ylim(0,1)
    plt.xlabel(str(param))
    plt.show()

def plotAccuraciesToParam(paramstr, param, acclist):
    colors = ['r','b']
    for i, acc in enumerate(acclist):
        plt.plot(param, acc, colors[i] + 'o-')
    plt.ylabel('Prediction Accuracy')
    plt.xlabel(paramstr)
    plt.ylim(0, 1.05)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.05, 0.1))
    plt.show()

def plotDistanceFrequency(A, dist):
    dfmap = {}
    for row in A:
        for e in row:
            if e in dfmap:
                dfmap[e] += 1
            else:
                dfmap[e] = 1
    
    distances = list(dfmap.keys())
    freq = []
    for d in distances:
        freq.append(dfmap[d])
    
    plt.plot(distances, freq, 'bo')
    plt.ylabel('Frequency')
    plt.xlabel(str(dist))
    plt.show()
