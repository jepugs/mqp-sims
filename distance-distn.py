#from joblib import Parallel, delayed
from math import ceil
from random import random
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from metrics import as_adj, dsd_mat, spd_mat
from sim_cwba import cwba_graph
import completeGraphs as ncc


# these functions only work under the assumption that the nodes of a graph are
# labeled by the integers 0 through n-1.

# get all shortest path distances in the graph
def spdists(G):
    A = as_adj(G)
    spds = spd_mat(A)
    res = []
    for u in range(0, A.shape[0]):
        for v in range(u + 1, A.shape[0]):
            res.append(spds[u][v])
    return res

# get a list of all shortest path distances in the graph
def dsdists(G):
    A = as_adj(G)
    ds = dsd_mat(A)
    res = []
    for u in range(0, A.shape[0]):
        for v in range(u + 1, A.shape[0]):
            res.append(ds[u,v])
    return res

def class_dists(G, truth_tab):
    """Returns a three-column data frame where the first column is all shortest path
    distances in G and the second is corresponding DSDs. The third column is 0
    if the two vertices have the same class and 1 otherwise. Paths of length 0
    are skipped (when source node == target node).

    Parameters
    ----------
    G : networkx.Graph or numpy.ndarray
    truth_tab : dict
        A dictionary of labels keyed by each vertex in G

    Returns
    -------
    df : DataFrame
    """

    A = as_adj(G)
    # total number of distances
    n_rows = int((A.shape[0] * (A.shape[0] - 1)) / 2)
    # numpy table to hold the result before we stick it in the data frame
    tab = np.empty((n_rows,3), dtype=np.float64)
    dsds = dsd_mat(A)
    spds = spd_mat(A)

    col = 0
    for u,v in combinations(range(A.shape[0]), r=2):
        if u == v:
            continue
        tab[col,0] = spds[u,v]
        tab[col,1] = dsds[u,v]
        tab[col,2] = 0 if truth_tab[u] == truth_tab[v] else 1
        col += 1

    return pd.DataFrame(tab, columns=['spd','dsd', 'class_diff'])


def class_dists_many(iterable):
    """Vectorized version of class_dists"""
    return pd.concat(map(lambda x: class_dists(x[0], x[1]), iterable), axis=0)


def class_degrees(G, truth_tab):
    """Computes average degrees of all nodes in G, as well as average number of
    same-class and different-class neighbors.

    Parameters
    ----------
    G : networkx.Graph
    truth_tab : dict
        A dictionary of labels keyed by each vertex in G

    Returns
    -------
    degrees : tuple
        A length-3 tuple consisting of the average degree, the average number of
        same-class neigbors, and the average number of different-class
        neighbors, in that order.

    """

    # running sum of same neighbors
    acc_same = 0
    # running sum of different neighbors
    acc_diff = 0
    for u in G:
        for v in G.neighbors(u):
            if truth_tab[u] == truth_tab[v]:
                acc_same += 1
            else:
                acc_diff += 1

    same = acc_same / G.order()
    diff = acc_diff / G.order()

    return same + diff, same, diff

# gen should generate tuples (graph,truth_tab)
def class_degrees_avg(iterable):
    """Like class_degrees, but computes averages over many (graph,truth_table) 
    pairs.
    """

    vals = pd.DataFrame(map(lambda x: class_degrees(x[0],x[1]), iterable))
    return tuple(vals.mean())

def gen_inputs(count, func, *args, **kwargs):
    """Returns a generator that yields count inputs created by calling func with 
    the parameters provided.
    """
    for i in range(count):
        yield func(*args, **kwargs)

def class_hists(name, gen, file_prefix=None, file_ext='.png', show=True):
    def fname(s):
        if file_prefix == None:
            return None
        return file_prefix + s + file_ext
    df = class_dists_many(gen)
    df_same = df.loc[df.class_diff==0]
    df_diff = df.loc[df.class_diff==1]

    plot_hist('Same class distances for %s' % name, list(df_same.spd), 
              list(df_same.dsd), filename=fname('_same'), show=show)
    plot_hist('Different class distances for %s' % name, list(df_diff.spd),
              list(df_diff.dsd), filename=fname('_diff'), show=show)
    plot_hist('All distances for %s' % name, list(df.spd), list(df.dsd),
              filename=fname('_all'), show=show)

    return df, df_same, df_diff


def compute_and_plot_hist(title, graph_gen, bins=None, filename=None):
    ds = [] 
    dsds = []

    N = 0
    for G in graph_gen:
        N += 1
        ds.extend(spdists(G))
        dsds.extend(dsdists(G))

    plot_hist(title,ds,dsds,filename=filename)


def plot_hist(title, ds, dsds, filename=None, show=True):
    mdsd = max(dsds)
    md = max(ds)
    maxBin = ceil(max(md, mdsd))
    bins = list(map(lambda x: x * maxBin * 0.01, range(100)))

    plt.hist((dsds,ds), bins=bins, density=True,
             label=("DSD","Shortest Path"), histtype='step')
    plt.legend(prop={'size':10})
    plt.title(title)
    plt.xlabel('Distance')
    plt.ylabel('Fraction of all distances')
    if filename != None:
        plt.savefig(filename)
    if show:
        plt.show()
        plt.close()


rrl = nx.connected_watts_strogatz_graph(500, 10, 0)
ws01 = nx.connected_watts_strogatz_graph(500, 10, 0.01)
ws10 = nx.connected_watts_strogatz_graph(500, 10, 0.10)
ws20 = nx.connected_watts_strogatz_graph(500, 10, 0.20)
ws50 = nx.connected_watts_strogatz_graph(500, 10, 0.50)


def main():
    pass


## These functions create generators for plain networkx graphs/adjacency tables

def gen_ws(count, n, k, p):
    for i in range(count):
        G = nx.connected_watts_strogatz_graph(n, k, p)
        yield G

# order is 2n, nodes are deleted within classes w/ probability q, then added
# between classes with probability p
def gen_ncc(count, n, p, q):
    for i in range(count):
        G = ncc.construct(n, p, q)
        yield G[0]

def gen_cwba(count, n, m, rho, labels=[0,1]):
    for i in range(count):
        G = cwba_graph(n,m,rho,labels)
        yield G[0]

