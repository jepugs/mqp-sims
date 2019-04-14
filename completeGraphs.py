import networkx as nx
import itertools
import random

from itertools import cycle,chain,repeat

import numpy as np


def add_ncc_noise(G, p, q):
    for u, v in itertools.combinations(G.nodes(), r=2):
        if u == v:
            pass
        if G.has_edge(u,v):
            if random.random() < q:
                G.remove_edge(u,v)
        elif random.random() < p:
            G.add_edge(u,v)

    return G

def removeEdges(G, q):
    edges = list(G.edges()).copy()
    for e in edges:
        u, v = e
        if random.random() < q:
            G.remove_edge(u, v)
    return G

def addEdges(G, p):
    # class1 = range(int(len(G)/2))
    # class2 = range(int(len(G)/2)), int(len(G)
    # for u,v in itertools.product(class1, class2):
    #     if random.random() < p:
    #         G.add_edge(u, v)
    # return G

    for u, v in itertools.combinations(G.nodes(), r=2):
        sizeG = len(G)/2
        if not ((u < sizeG and v < sizeG) or (u >= sizeG and v >= sizeG)):
            if random.random() < p:
                G.add_edge(u, v)
    return G

def addHubs(G, r):
    Gcopy = G.copy()
    for i in range(r):
        newhub = i+len(G)
        G.add_node(newhub)
        # try using np.random.choice
        for v in Gcopy: # try just G
            if random.random() < 0.7:
                G.add_edge(newhub, v)
    return G


def construct(n, p, q):
    # Create complete graphs of size n
    G = nx.complete_graph(n)
    H = nx.complete_graph(n)
    GH = nx.disjoint_union(G, H)
    # Remove edges within disjoint subgraphs
    GH = removeEdges(GH, q)
    # Add edges between the disjoint subgraphs
    GH = addEdges(GH, p)
    # Get the largest connected component
    GH = max(nx.connected_component_subgraphs(GH), key=len)
    # Create a truth table
    truth = {}
    for v in GH:
        if v < len(G):
            truth[v] = 'r'
        else:
            truth[v] = 'b'
    return GH, truth

def constructWithHubs(n, p, q, r):
    G, truth = construct(n, p, q)
    hubindex = len(G)
    G = addHubs(G, r)
    for hub in range(hubindex, len(G)):
        if random.random() < 0.5:
            truth[hub] = 'r'
        else:
            truth[hub] = 'b'
    return G, truth


def bfs_cc(A, source):
    visited = [False] * A.shape[0]
    visited[source] = True
    queue = [source]
    while queue:
        queue.pop(0)
        adj_i = np.nonzero(np.asarray(A[source,:]))
        for i in adj_i:
            if visited[i] == False:
                queue.append(i)
                visited[i] = True
    return visited

def lcc(A):
    visited = [False] * A.shape[0]
    lcc = []
    source = 0
    while False in visited:
        visit_i = bfs_cc(A, source)
        visited[visit_i] = True
    
    

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
