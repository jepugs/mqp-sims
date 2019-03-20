import networkx as nx
import itertools
import random


def removeEdges(G, q):
    edges = list(G.edges()).copy()
    for e in edges:
        u, v = e
        if random.random() < q:
            G.remove_edge(u, v)
    return G

def addEdges(G, p):
    # class1 = range(len(G)/2)
    # class2 = range(len(G)/2, len(G))
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
            if random.random() < 0.8:
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
