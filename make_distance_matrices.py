from itertools import count
import networkx as nx
import numpy as np
from metrics import *


#### Coauthor dataset
#dataset = 'coauthorGiant'

# this loads an adjacency matrix
#adj = np.matrix(np.loadtxt('./' + dataset + 'Adj.txt', delimiter=' '))

# used diff to prove these files were the same, verifying the data format
#np.savetxt('testout.txt', adj, fmt='%d',  delimiter=' ', newline='\r\n')


####################################################################################################

#### email-Eu-core dataset

# this loads an edge list (we pretend that the graph is undirected)
dataset = 'email-Eu-core'

edges = np.loadtxt('./email-Eu-core.txt', delimiter=' ')
# get the largest connected component
G = max(nx.connected_component_subgraphs(nx.from_edgelist(edges)), key=len)
# rebuild the truth table
email_truth = dict(np.loadtxt('email-Eu-core-department-labels.txt', delimiter=' '))
truth_fixed = {}
for (i,v) in zip(count(),G):
    truth_fixed[i] = email_truth[v]

# convert to a numpy array and write it out
truth_array = np.array([(k,truth_fixed[k]) for (k) in truth_fixed.keys()], dtype=np.int)

# write fixed truth table
np.savetxt('email-Eu-coreLabels.txt', truth_array, fmt='%d', delimiter=' ')

# relabel nodes to match the new truth table
G = nx.convert_node_labels_to_integers(G)
adj = nx.to_numpy_matrix(G)



####################################################################################################

#### write out results

# write out a space-delimited file for each distance matrix
np.savetxt(dataset + 'DSD.txt', dsd_mat(adj), delimiter=' ', newline='\n')
np.savetxt(dataset + 'SPD.txt', spd_mat(adj), fmt='%d', delimiter=' ', newline='\n')
np.savetxt(dataset + 'RD.txt', rd_mat(adj), delimiter=' ', newline='\n')
