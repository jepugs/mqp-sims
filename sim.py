from sklearn.model_selection import KFold

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

def runsim(truth, censorP, votefn, metric, avg_runs=10, **kwargs):
    correct = 0
    num_censor = int(np.floor(censorP * len(truth.keys())))
    for i in range(avg_runs):
        censored = voting.censor(truth, num_censor)
        predicted_correct = votefn(censored, truth, metric, **kwargs)
        correct += predicted_correct
    return correct, avg_runs*num_censor


def runsim_cv(truth, votefn, metric, n_folds=5, shuffle=True, **kwargs):
    """Cross-validated version of runsim. By default, data is shuffled before splitting. extra kwargs
are passed to the voting function.

    """
    correct = 0
    
    kf = KFold(n_splits=n_folds, shuffle=shuffle)
    for train, test in kf.split(truth):
        # create the training dictionary
        predicted_correct = votefn(test, truth, metric, **kwargs)
        correct += predicted_correct
    return correct, len(truth)
