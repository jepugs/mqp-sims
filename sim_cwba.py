from itertools import cycle, islice
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def cwba_graph(n, m, rho, labels=[0,1], seed=None):
    """Returns a random graph according to the class-weighted Barabási–Albert
    preferential attachment model. Note that this version uses an empty graph of
    order m for G_0.

    A graph of $n$ nodes is grown by attaching new nodes each with $m$ edges
    that are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    rho : float
        Relative weighting coefficient for same-class attachment. That is,
        classes with the same label as a new node have rho times the connection
        probability that they would have if their label were different.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : Graph
    truth_tab : Dictionary of all classes

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m < n``.
    """

    if m < 1 or m >= n:
        raise nx.NetworkXError("CWBA network must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n))

    # Add m initial nodes (m0 in barabasi-speak)
    G = nx.empty_graph(m)

    # degree of each node
    degrees = np.zeros(n)
    # probability distribution for drawing edges
    pmf = np.zeros(n)
    # assign each vertex in m a label by cycling the given ones
    truth_tab = {}
    for v, l in zip(G, cycle(labels)):
        truth_tab[v] = l

    # We start by connecting node m to all existing nodes (forming a star)
    G.add_edges_from(zip(cycle([m]), range(m)))
    # assign degrees for the starting graph
    for i in range(m):
        degrees[i] = 1
    degrees[m] = m
    # assign a label to the new node
    cur_label = np.random.choice(labels)
    truth_tab[m] = cur_label

    # add remaining nodes
    source = m+1
    while source < n:
        # randomly pick a label for the new node
        cur_label = np.random.choice(labels)
        truth_tab[source] = cur_label

        # compute pmf
        for v in range(source):
            if truth_tab[v] == cur_label:
                # same label as the new node => greater probability
                pmf[v] = rho*degrees[v]
            else:
                pmf[v] = degrees[v]
        # normalize
        pmf /= np.sum(pmf)

        # sample nodes without replacement to add to the graph
        targets = np.random.choice(list(islice(G, source)), size=m,
                                   replace=False, p=pmf[:source])

        # update the degrees list
        degrees[source] = m
        for t in targets:
            degrees[t] += 1

        # Add edges to m nodes from the source. add_edges_from will
        # automatically create new nodes
        G.add_edges_from(zip([source] * m, targets))

        source += 1
    return G,truth_tab


# used to test CWBA-- we expect to see a power law here
def plotDegrees(G, graph_title='', filename=None):
    plt.plot(list(map(G.degree,range(G.order()))))
    plt.title('Degree Distribution for %s' % graph_title)
    plt.ylabel('degree')
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()
