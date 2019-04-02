import networkx as nx
import numpy as np

import voting

"""
Run a simulation
    @param G: networkx graph
    @param truth: dictionary containing label information of vertices in G
    @param censorP: proportion of vertices in G to censor labels
    @param vote: function for the voting algorithm
    @param metric: numpy symmetric matrix of distances between vertices in G
    @param avgRuns: number of times to run prediction to get an average accuracy
    
    @return: tuple with the average number of correctly predicted labels and the total number of predicted labels
"""

def runsim(truth, censorP, votefn, metric, avgRuns=10):
    avg_correct = 0
    total = 0
    for i in range(avgRuns):
        censored = voting.censor(truth, censorP)
        predicted_correct, total = votefn(censored, truth, metric)
        avg_correct += predicted_correct
    return avg_correct/avgRuns, total
