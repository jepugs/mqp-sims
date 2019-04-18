import numpy as np
from itertools import cycle,chain,repeat

# complete components adjacency matrix
def cc_graph_adj(n):
    o = np.ones((n,n)) - np.eye(n)
    z = np.zeros((n,n))
    top = np.concatenate((o,z), axis=1)
    bot = np.concatenate((z,o), axis=1)
    return np.concatenate((top,bot), axis=0)

# add ncc noise to an adjacency matrix
def add_adj_noise(A, p, q):
    # random matrix with A's dimensions
    r0 = np.random.rand(A.shape[0], A.shape[1])
    # make r0 symmetric
    r = (r0 + r0.T)/2
    # Adjacency matrix of the graph complement. Subtract I so we avoid cycles
    Ac = 1 - A - np.eye(A.shape[0])
    # compare random values to p to decide when to add edges
    adds = r < p
    # decide when to remove edges (if they exist)
    rems = r < q
    return A + np.multiply(Ac,adds) - np.multiply(A,rems)

# construct an ncc adjacency matrix and truth table
def construct_adj(n, p, q):
    A0 = cc_graph_adj(n)
    A = add_adj_noise(A0,p,q)
    labels = chain(repeat(0,n), repeat(1,n))
    truth = dict(zip(range(2*n), labels))
    return A, truth
