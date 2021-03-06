import networkx as nx
import numpy as np
import itertools
from numpy import linalg as LA
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path

def as_adj(G):
    """Ensure that G is an adjacency matrix, converting it from an networkx graph
    if necessary"""
    A = None
    if isinstance(G,np.ndarray):
        A = G
    else:
        A = nx.to_numpy_matrix(G)
    return A

"""
Compute a symmetric matrix of all DSD values
Credit for this algorithm goes to Enrico Maiorino, see https://github.com/reemagit/DSD
    @param G: networkx graph or numpy.ndarray
    
    @return: symmetric matrix of all DSD values
"""

def dsd_mat(G):
    A = as_adj(G)
    n = A.shape[0]
    degree = A.sum(axis=1)
    p = A / degree
    pi = degree / degree.sum()
    return squareform(pdist(LA.inv(np.eye(n) - p - pi.T), metric='cityblock'))

def spd_mat(G):
    A = as_adj(G)
    return shortest_path(A, directed=False, unweighted=True)

def rd_mat(G):
    A = as_adj(G)
    D = np.zeros(A.shape)
    for i in range(A.shape[0]):
        D[i,i] = np.sum(A[i,:])
    L = D - A
    Lp = L +  (1/A.shape[0]) * np.ones(L.shape)
    Lp_inv = np.linalg.inv(Lp)
    Lp_inv_diag = np.diag(Lp_inv)
    RD = np.add.outer(Lp_inv_diag, Lp_inv_diag) - 2*Lp_inv
    return RD
