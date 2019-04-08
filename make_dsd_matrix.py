import numpy as np
import calcDSD
from metrics import *

dataset = 'coauthorGiant'

# this loads the matrix
adj = np.matrix(np.loadtxt('./' + dataset + 'Adj.txt', delimiter=' '))

# used diff to prove these files were the same, verifying the data format
#np.savetxt('testout.txt', adj, fmt='%d',  delimiter=' ', newline='\r\n')

# write out a space-delimited file for each distance matrix
np.savetxt(dataset + 'DSD.txt', dsd_mat(adj), delimiter=' ', newline='\n')
np.savetxt(dataset + 'SPD.txt', spd_mat(adj), fmt='%d', delimiter=' ', newline='\n')
np.savetxt(dataset + 'RD.txt', rd_mat(adj), delimiter=' ', newline='\n')
