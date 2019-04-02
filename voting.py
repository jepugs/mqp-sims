import numpy as np

def censor(truth, censorP):
    """
    Censor labels

    Parameters
    ----------
    truth : dict
        Dictionary of vertices and their labels
    censorP : float
        Proportion of vertices to censor (between 0 and 1)

    Returns
    -------
    censored : List of censored vertices
    """
    if censorP > 1:
        censorP = 1
    if censorP < 0:
        censorP = 0
    num_censor = int(np.floor(censorP * len(truth.keys())))
    censored = np.random.choice(list(truth.keys()), size=num_censor, replace=False)
    return censored

def random_vote(censored, truth, metric):
    """
    Random voting of labels

    Parameters
    ----------
    censored : list
        List of censored vertices
    truth : dict
        Dictionary of vertices and their labels
    metric : numpy matrix
        Matrix of distances between pairs of vertices

    Returns
    -------
    predicted_correct : Number of correctly predicted labels
    predicted_total : Total number of predicted labels
    """
    predicted_total = len(censored)
    predicted_correct = 0
    for i in range(predicted_total):
        if np.random.random() < 0.5:
            predicted_correct += 1
    return predicted_correct, predicted_total


def knn_weighted_majority_vote(censored, truth, metric, k=20):
    """
    Weighted majority vote using k-nearest neighbors

    Parameters
    ----------
    censored : list
        List of censored vertices
    truth : dict
        Dictionary of vertices and their labels
    metric : numpy matrix
        Matrix of distances between pairs of vertices
    k : int
        Number of nearest neighbors to consider for voting (default is 20)

    Returns
    -------
    predicted_correct : Number of correctly predicted labels
    predicted_total : Total number of predicted labels
    """
    predicted_total = len(censored)
    predicted_correct = 0
    if k > metric.shape[0] - 1: # if there are less than `k` vertices
        k = metric.shape[0] - 1
    for i in range(predicted_total):
        current_vertex = censored[i]
        row = np.asarray(metric[current_vertex,:]) # row of `metric` matrix
        row = row[np.nonzero(row)] # remove 0's from row
        knn_i_list = list(np.argpartition(row, k))[:k+1] # list of indices of `k`-nearest neighbors
        num_k_val = 0
        neighbors = []
        rand_sample = []
        for j in range(len(knn_i_list)): # count the values in row equal to `k`th value of the row
            if row[knn_i_list[j]] == row[knn_i_list[k]]:
                num_k_val += 1
            else:
                neighbors.append(knn_i_list[j])
        for l in range(len(row)): # get all values in row equal to `k`th value of the row
            if row[l] == row[knn_i_list[k]]:
                rand_sample.append(l)
        neighbors.extend(list(np.random.choice(rand_sample, num_k_val))) # get a random sample of these values
        votes = {}
        pop_tally = 0
        pop_votes = []
        pop_label = ""
        for n in neighbors: # record votes of all `k` neighbors
            if truth[n] in votes:
                votes[truth[n]] += 1/row[n]
            else:
                votes[truth[n]] = 1/row[n]
        for v in votes: # tally votes
            if votes[v] > pop_tally:
                pop_tally = votes[v]
                pop_votes = [v]
            elif votes[v] == pop_tally:
                pop_votes.append(v)
        pop_index = np.random.randint(len(pop_votes))
        pop_label = pop_votes[pop_index] # choose the popular label
        if pop_label == truth[current_vertex]:
            predicted_correct += 1
    return predicted_correct, predicted_total


def eb_weighted_majority_vote(censored, truth, metric, epsilon=3.0):
    """
    Weighted majority vote using epsilon-ball radius neighbors

    Parameters
    ----------
    censored : list
        List of censored vertices
    truth : dict
        Dictionary of vertices and their labels
    metric : numpy matrix
        Matrix of distances between pairs of vertices
    epsilon : float
        Radius of ball for considering neighbors to vote (default is 3.0)

    Returns
    -------
    predicted_correct : Number of correctly predicted labels
    predicted_total : Total number of predicted labels
    """
    predicted_total = len(censored)
    predicted_correct = 0
    if len(np.nonzero(metric <= epsilon)[0]) == 0: # if epsilon-ball radius is too small, set to average distance
        epsilon = np.mean(metric)
    for i in range(predicted_total):
        current_vertex = censored[i]
        row = np.asarray(metric[current_vertex,:]) # row of `metric` matrix
        row = row[np.nonzero(row)] # remove 0's from row
        neighbors = np.nonzero(row <= epsilon)[0] # get neighbors within epsilon-ball radius
        votes = {}
        pop_tally = 0
        pop_votes = []
        pop_label = ""
        for n in neighbors: # record votes of all neighbors
            if truth[n] in votes:
                votes[truth[n]] += 1/row[n]
            else:
                votes[truth[n]] = 1/row[n]
        for v in votes: # tally votes
            if votes[v] > pop_tally:
                pop_tally = votes[v]
                pop_votes = [v]
            elif votes[v] == pop_tally:
                pop_votes.append(v)
        if len(pop_votes) == 0: # if no neighbors voted, consider vote predicted false
            pop_label = ""
        else:
            pop_index = np.random.randint(len(pop_votes))
            pop_label = pop_votes[pop_index] # choose the popular label
        if pop_label == truth[current_vertex]:
            predicted_correct += 1
    return predicted_correct, predicted_total
