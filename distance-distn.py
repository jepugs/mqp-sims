#from joblib import Parallel, delayed
from math import ceil
from random import random
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from mydsd import dsdMat
from sim_cwba import cwba_graph
import sim as ncc


# these functions only work under the assumption that the nodes of a graph are
# labeled by the integers 0 through n-1.

# get a list of all shortest path distances in the graph
def spdists(G):
    ds = dict(nx.shortest_path_length(G))
    res = []
    for u in range(0, len(G.nodes)):
        for v in range(u + 1, len(G.nodes)):
            res.append(ds[u][v])
    return res

# get a list of all shortest path distances in the graph
def dsdists(G):
    ds = dsdMat(G)
    res = []
    for u in range(0, len(G.nodes)):
        for v in range(u + 1, len(G.nodes)):
            res.append(ds[u,v])
    return res

def gen_ws(count, n, k, p):
    for i in range(count):
        G = nx.connected_watts_strogatz_graph(n, k, p)
        yield G

# order is 2n, nodes are deleted within classes w/ probability q, then added
# between classes with probability p
def gen_ncc(count, n, p, q):
    for i in range(count):
        G = construct(n, p, q)
        yield G

def compute_and_plot(name, graph_gen, bins=None, filename=None):
    ds = []
    dsds = []

    N = 0
    for G in graphGen:
        N += 1
        ds.extend(spdists(G))
        dsds.extend(dsdists(G))

    plotHist(name,N,ds,dsds,filename=filename)


def plot_hist(name, N, ds, dsds, filename=None):
    mdsd = max(dsds)
    md = max(ds)
    bins = ceil(max(md, mdsd))

    plt.hist((dsds,ds), bins=bins, density=True,
             label=("DSD","Shortest Path"), histtype='step')
    plt.legend(prop={'size':10})
    plt.title('DSD and SPD Distributions on %s (N=%d)' % ( name, N))
    plt.xlabel('Distance')
    plt.ylabel('Fraction of all pairwise distances')
    plt.show()


rrl = nx.connected_watts_strogatz_graph(500, 10, 0)
ws01 = nx.connected_watts_strogatz_graph(500, 10, 0.01)
ws10 = nx.connected_watts_strogatz_graph(500, 10, 0.10)
ws20 = nx.connected_watts_strogatz_graph(500, 10, 0.20)
ws50 = nx.connected_watts_strogatz_graph(500, 10, 0.50)


def main():
    pass
