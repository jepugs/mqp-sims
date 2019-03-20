import networkx as nx
import numpy as np

import censor
#import voting
#import metric
#import plot

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


def runsim(G, truth, censorP, vote, metric, avgRuns):
    avg_correct = 0
    total = 0
    for i in range(avgRuns):
        # Create a censored truth table from truth
        censored = censor.censor(truth, censorP)
        # Vote for censored labels
        predicted_correct, total = vote(G, censored, truth, metric)
        avg_correct += predicted_correct
    return avg_correct/avgRuns, total
