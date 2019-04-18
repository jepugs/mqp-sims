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

# add hubs to an adjacency matrix
def add_hubs(A, r, rp):
    n = A.shape[0]
    new_hub_degree = int(np.floor(n*rp))
    for i in range(r):
        new_hub_col = np.hstack((np.ones(new_hub_degree),np.zeros(n-new_hub_degree)))
        np.random.shuffle(new_hub_col)
        new_hub_col = np.hstack((new_hub_col,np.zeros(i)))
        new_hub_row = np.hstack((new_hub_col,np.zeros(1)))
        A = np.vstack((np.hstack((A,new_hub_col.reshape((-1,1)))),new_hub_row))
    return A

# construct an ncc adjacency matrix and truth table
def construct_adj(n, p, q, r, rp):
    A0 = cc_graph_adj(n)
    A1 = add_adj_noise(A0,p,q)
    A = add_hubs(A1, r, rp)
    labels = chain(repeat(0,n), repeat(1,n))
    truth = dict(zip(range(2*n), labels))
    hub_labels = np.hstack((np.zeros(int(np.floor(r/2))),np.ones(int(np.ceil(r/2)))))
    np.random.shuffle(hub_labels)
    truth.update(dict(zip(range(2*n,2*n+r), hub_labels)))
    return A, truth
