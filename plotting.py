import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_param_vs_accuracy(param_data, acc_data, paramstr):
    plt.plot(param_data, acc_data, label=paramstr, color='orange', marker='o', markersize=6)
    plt.title(paramstr + " vs Prediction accuracy")
    plt.ylabel('Prediction accuracy')
    plt.xlabel(paramstr)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.xlim(left=0)
    plt.show()

def plot_params_vs_accuracy(param_data, acc_data_list, paramstr):
    colors = ['#ffaf42', '#4a42ff']
    markers = ['.', '^']
    plt.title("TEST")
    plt.ylabel("Prediction accuracy")
    plt.xlabel(paramstr)
    plt.ylim(0, 1.05)
    plt.xlim(left=0)
    for i, acc_data in enumerate(acc_data_list):
        plt.plot(param_data, acc_data, label=paramstr, color=colors[i], marker=markers[i])
    plt.legend()
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
