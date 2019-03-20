import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import math

"""
Random voting
"""
def randomVote(G, censored, truth, metric):
    predicted = 0
    total = 0
    for v in censored:
        if random.random() < 0.5:
            predicted += 1
        total += 1
    return predicted, total

"""
Local unweighted vote
    @param G: graph
    @param neighbors: list of neighbors that vote
"""
def localVote(G, neighbors):
    popular_vote = ""
    votes = {}
    # Go through neighbors and tally votes
    for n in neighbors:
        label = G.nodes[n]['label']
        if not label == "":
            if label in votes:
                votes[label] += 1
            else:
                votes[label] = 1
    # If votes is empty, return empty label
    if votes:
        popular_vote = max(votes, key=votes.get)
    return popular_vote

# Majority vote
# @param G: graph
# @param radius: radius of neighbors to consider when voting
# @param A: numpy matrix of distances/similarities
# @param G_truth: ground truth graph
# @param names: list of names in order corresponding to matrix A
def majorityVote(G, radius, A, G_truth, names):
    total_predicted = 0
    total_predicted_correct = 0
    for i,n in enumerate(G): # Loop through nodes in G
        label = G.nodes(data='label')[n]
        if label == "": # If label needs to be predicted, vote
            neighbors = []
            for a_index,a in enumerate(A[i,:]): # Go through similarity values in A and get neighbors that will vote
                if a <= radius:
                    neighbors.append(names[a_index])
            pop_vote = localVote(G, neighbors) # Get label from popular vote
            if pop_vote == G_truth.nodes[n]['label']: # Check if predicted correctly
                total_predicted_correct += 1
            total_predicted += 1
    return total_predicted_correct, total_predicted
            
# Local unweighted vote
# @param G: graph
# @param neighbors: list of neighbors that vote
def localWeightedVote(G, neighbors, neighborWeights):
    popular_vote = ""
    votes = {}
    # Go through neighbors and tally votes
    for n,nweight in zip(neighbors, neighborWeights):
        label = G.nodes(data='label')[n]
        if label != "" and nweight != 0:
            if label in votes:
                votes[label] += 1/nweight
            else:
                votes[label] = 1/nweight
    # If votes is empty, return empty label
    if votes:
        popular_vote = max(votes, key=votes.get)
    return popular_vote

# Majority vote
# @param G: graph
# @param radius: radius of neighbors to consider when voting
# @param A: numpy matrix of distances/similarities
# @param G_truth: ground truth graph
# @param names: list of names in order corresponding to matrix A
def majorityWeightedVote(G, radius, A, G_truth, names):
    total_predicted = 0
    total_predicted_correct = 0
    for i,n in enumerate(G): # Loop through nodes in G
        label = G.nodes(data='label')[n]
        if label == "": # If label needs to be predicted, vote
            neighbors = []
            neighborWeights = []
            for a_index in range(A.shape[0]): # Go through similarity values in A and get neighbors that will vote
                a = A[i,a_index]
                if a <= radius:
                    neighbors.append(names[a_index])
                    neighborWeights.append(a)
            pop_vote = localWeightedVote(G, neighbors, neighborWeights) # Get label from popular vote
            if pop_vote == G_truth.nodes(data='label')[n]: # Check if predicted correctly
                total_predicted_correct += 1
            total_predicted += 1
    return total_predicted_correct, total_predicted

"""
Weighted Majority Vote
using k-nearest neighbors
    @param G: networkx graph
    @param censored: array of censored vertices in G
    @param truth: dictionary of label data for vertices in G
    @param metric: numpy symmetric matrix with pairwise distances of vertices in G

    @return: tuple with the number of correctly predicted labels and the total number of predicted labels
"""

# TODO: add for f1-score
def weightedMajorityVote(G, censored, truth, metric):
    k = 20
    predicted_correct = 0
    predicted_total = 0
    for v in censored:
        #v = int(v)
        neighbor_i = list(np.argpartition(list(metric[v,:]), k))[:k+1]
        neighbor = list(map(lambda x: metric[v, x], neighbor_i))
        votes = {}
        pop_vote = ""
        pop_votes = []
        pop_votes_val = 0
        for n in neighbor_i:
            if metric[v, n] == 0:
                continue
            if truth[n] in votes:
                votes[truth[n]] += 1/metric[v, n]
            else:
                votes[truth[n]] = 1/metric[v, n]
        for label, voteval in votes.items():
            if voteval > pop_votes_val:
                pop_votes_val = voteval
                pop_votes = [label]
            elif voteval == pop_votes_val:
                pop_votes.append(label)
        if len(pop_votes) > 1:
            index = random.randint(0, len(pop_votes) - 1)
            pop_vote = pop_votes[index]
        else:
            pop_vote = pop_votes[0]
        if pop_vote == truth[v]:
            predicted_correct += 1
        predicted_total += 1
        
    return predicted_correct, predicted_total


def weightedMajorityVote2(G, censored, truth, metric):
    k = 20
    predicted_correct = 0
    predicted_total = 0
    for v in censored:
        #print("voting for: ", v)
        neighbor_i = list(np.argpartition(metric[v,], k + 1))[:k + 2]
        neighbor = list(map(lambda x: metric[v, x], neighbor_i))
        pop_vote = 0
        pop_label = 'Mr. Hi'
        for n in neighbor_i:
            if metric[v, n] == 0:
                continue
            if truth[n] == 'Mr. Hi':
                pop_vote += 1 / metric[v, n]
            else:
                pop_vote -= 1 / metric[v, n]
        if pop_vote > 0:
            pop_label = 'Mr. Hi'
        elif pop_vote < 0:
            pop_label = 'Officer'
        else:
            pop_label = 'Mr. Hi' if random.random() > 0.5 else 'Officer'
        if pop_label == truth[v]:
            predicted_correct += 1
        predicted_total += 1

    return predicted_correct, predicted_total
