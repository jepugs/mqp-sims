import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from itertools import count
from pathlib import Path
import sim
import metrics
import voting

import plotting

import completeGraphs as cg



"""
Test for running multiple test cases of complete graphs
"""


def suite_completeGraphs(n, q, testfn, censorP, vote, metric, avgRuns):
    p = Decimal(str(0.0))
    acc = []
    param = []
    while p <= 1:
        testacc = testfn(n, p, q, censorP, vote, metric, avgRuns)
        acc.append(testacc)
        param.append(p)
        p += Decimal(str(0.25))
        print(testacc)
    return acc, param


"""
Test case for complete graphs
"""


def test_completeGraphs(n, p, q, censorP, vote, metric, avgRuns):
    G, truth = cg.construct(n, p, q)
    correct, total = sim.runsim(truth, censorP, vote, metric(G), avgRuns)
    return correct / total


def runtest_completeGraphs():
    n = 200
    q = 0.5
    censorP = 0.3
    avgRuns = 10
    dsdacc, dsdparam = suite_completeGraphs(n, q, test_completeGraphs, censorP, voting.knn_weighted_majority_vote, metrics.dsd_mat, avgRuns)
    spdacc, spdparam = suite_completeGraphs(n, q, test_completeGraphs, censorP, voting.knn_weighted_majority_vote, metrics.spd_mat, avgRuns)
    plotting.plot_params_vs_accuracy(spdparam, [spdacc, dsdacc], "p")
    return


#runtest_completeGraphs()

def suite_completeGraphsWithHubs(n, q, r, testfn, censorP, vote, metric, avgRuns):
    p = Decimal(str(0.0))
    acc = []
    param = []
    while p <= 1:
        testacc = testfn(n, p, q, r, censorP, vote, metric, avgRuns)
        acc.append(testacc)
        param.append(p)
        p += Decimal(str(0.025))
        print(testacc)
    return acc, param

def test_completeGraphsWithHubs(n, p, q, r, censorP, vote, metric, avgRuns):
    G, truth = cg.constructWithHubs(n, p, q, r)
    correct, total = sim.runsim(G, truth, censorP, vote, metric(G), avgRuns)
    return correct / total

def runtest_completeGraphsWithHubs():
    n = 200
    q = 0.5
    r = 100
    censorP = 0.3
    avgRuns = 100
    dsdacc, dsdparam = suite_completeGraphsWithHubs(n, q, r, test_completeGraphsWithHubs, censorP, voting.weightedMajorityVote, metrics.dsdMat, avgRuns)
    spdacc, spdparam = suite_completeGraphsWithHubs(n, q, r, test_completeGraphsWithHubs, censorP, voting.weightedMajorityVote, metrics.spdMat, avgRuns)
    plotting.plotAccuraciesToParam("p", dsdparam, [dsdacc, spdacc])
    return

#runtest_completeGraphsWithHubs()


def test_karateClubGraph():
    G = nx.karate_club_graph()
    truth = {}
    for v in G:
        truth[v] = G.nodes[v]['club']
    censorP = 0.3
    avgRuns = 1000
    #nx.draw_networkx(G)
    #plt.show()
    dsdcorrect, dsdtotal = sim.runsim(G, truth, censorP, voting.weightedMajorityVote, metrics.dsdMat(G), avgRuns)
    print(dsdcorrect)
    print(dsdtotal)
    spdcorrect, spdtotal = sim.runsim(G, truth, censorP, voting.weightedMajorityVote, metrics.spdMat(G), avgRuns)
    print(spdcorrect)
    print(spdtotal)
    print("DSD: ", dsdcorrect/dsdtotal)
    print("SPD: ", spdcorrect/spdtotal)
    return

#test_karateClubGraph()


def test_coauthorship():
    """
    http://konect.uni-koblenz.de/networks/com-dblp
    """
    fname = "../../Data/com-dblp.ungraph.txt/com-dblp.ungraph.txt"
    f = open(fname, 'rb')
    G = nx.read_edgelist(f, comments='#', delimiter=None, nodetype=int)
    #G = max(nx.connected_component_subgraphs(G), key=len)
    f.close()
    #print(len(G))

    truth = {}
    f2name = "../../Data/com-dblp.top5000.cmty.txt/com-dblp.top5000.cmty.txt"
    f2 = open(f2name, 'rb')
    linenum = 0
    for line in f2.readlines():
        if linenum > 500:
            break
        line = line.decode('UTF-8')
        community = line.rstrip().split('\t')
        for c in community:
            truth[c] = linenum
        linenum += 1
    #print(len(truth))
    f2.close()

    print(len(G))
    print(len(truth.keys()))
    for v in (G.copy()):
        if not(str(v) in truth):
            G.remove_node(v)
    G = max(nx.connected_component_subgraphs(G), key=len)
    print(len(G))

    mapping = {}
    vnum = 0
    for node in G.copy():
        mapping[node] = vnum
        vnum += 1
    nx.relabel_nodes(G, mapping, copy=False)
    newtruth = {}
    for m in mapping:
        newtruth[mapping[m]] = truth[str(m)]
    print(len(newtruth))

    censorP = 0.3
    avgRuns = 100

    dsdcorrect, dsdtotal = sim.runsim(G, newtruth, censorP, voting.weightedMajorityVote, metrics.dsdMat(G), avgRuns)
    print(dsdcorrect)
    print(dsdtotal)
    spdcorrect, spdtotal = sim.runsim(G, newtruth, censorP, voting.weightedMajorityVote, metrics.spdMat(G), avgRuns)
    print(spdcorrect)
    print(spdtotal)
    print("DSD: ", dsdcorrect/dsdtotal)
    print("SPD: ", spdcorrect/spdtotal)
    return
    
    

#test_coauthorship()

def test_emaileucore():
    fname = "../../Data/email-Eu-core/email-Eu-core.txt"
    f = open(fname, 'rb')
    G = nx.read_edgelist(f, comments='#', delimiter=None, nodetype=int)
    print(len(G))
    G = max(nx.connected_component_subgraphs(G), key=len)
    print(len(G))
    f.close()

    truth = {}
    f2name = "../../Data/email-Eu-core/email-Eu-core-department-labels.txt"
    f2 = open(f2name, 'rb')
    linenum = 0
    for line in f2.readlines():
        line = line.decode('UTF-8')
        nodedept = line.rstrip().split(' ')
        truth[nodedept[0]] = nodedept[1]
        linenum += 1
    print(len(truth))
    f2.close()

    truthcopy = list(truth.keys())
    for t in truthcopy:
        if not(G.has_node(int(t))):
            del truth[t]
    print(len(truth.keys()))

    mapping = {}
    vnum = 0
    for node in G.copy():
        mapping[node] = vnum
        vnum += 1
    nx.relabel_nodes(G, mapping, copy=False)
    newtruth = {}
    for m in mapping:
        newtruth[mapping[m]] = truth[str(m)]
    print(len(newtruth))

    censorP = 0.3
    avgRuns = 100

    dsdcorrect, dsdtotal = sim.runsim(G, newtruth, censorP, voting.weightedMajorityVote, metrics.dsdMat(G), avgRuns)
    print(dsdcorrect)
    print(dsdtotal)
    spdcorrect, spdtotal = sim.runsim(G, newtruth, censorP, voting.weightedMajorityVote, metrics.spdMat(G), avgRuns)
    print(spdcorrect)
    print(spdtotal)
    print("DSD: ", dsdcorrect/dsdtotal)
    print("SPD: ", spdcorrect/spdtotal)
    return

#test_emaileucore()


# coauthorship graph files (not included in repo)
coauthor_truth_filename = 'coauthorGiantSCORELabel.txt'
coauthor_dsd_filename = 'coauthorGiantDSD.txt'
coauthor_spd_filename = 'coauthorGiantSPD.txt'
coauthor_rd_filename = 'coauthorGiantRD.txt'

# save the matrices on multiple runs
def fexists(str):
    ''' check if a file exists based on its name '''
    return Path(str).is_file()

coauthor_truth = None
coauthor_dsd = None
coauthor_spd = None
coauthor_rd = None

coauthor_truth = np.loadtxt(coauthor_truth_filename, delimiter=' ') if \
    fexists(coauthor_truth_filename) and coauthor_truth is None else coauthor_truth
coauthor_dsd = np.loadtxt(coauthor_dsd_filename, delimiter=' ') if \
    fexists(coauthor_dsd_filename) and coauthor_dsd is None else coauthor_dsd
coauthor_spd = np.loadtxt(coauthor_spd_filename, delimiter=' ') if \
    fexists(coauthor_truth_filename) and coauthor_spd is None else coauthor_spd
coauthor_rd = np.loadtxt(coauthor_rd_filename, delimiter=' ') if \
    fexists(coauthor_truth_filename) and coauthor_rd is None else coauthor_rd


def test_coauthor_network_cv(n_folds=5, k=20):
    truth = dict(zip(count(), np.loadtxt(coauthor_truth_filename)))

    dsd_corr,dsd_total = sim.runsim_cv(truth, 
                                       voting.knn_weighted_majority_vote,
                                       coauthor_dsd, 
                                       n_folds=n_folds,
                                       k=k)
    print('DSD: %.2f' % (dsd_corr/dsd_total))

    spd_corr,spd_total = sim.runsim_cv(truth,
                                       voting.knn_weighted_majority_vote,
                                       coauthor_spd,
                                       n_folds=n_folds,
                                       k=k)
    print('SPD: %.2f' % (spd_corr/spd_total))

    rd_corr,rd_total = sim.runsim_cv(truth,
                                     voting.knn_weighted_majority_vote,
                                     coauthor_rd, 
                                     n_folds=n_folds,
                                     k=k)
    print('RD: %.2f' % (rd_corr/rd_total))

def test_coauthor_network(censor_rate=0.7, k=20):
    truth = dict(zip(count(), np.loadtxt(coauthor_truth_filename)))

    dsd_corr,dsd_total = sim.runsim(truth,
                                    censor_rate,
                                    voting.knn_weighted_majority_vote,
                                    coauthor_dsd, 
                                    k=k)
    print('DSD: %.2f' % (dsd_corr/dsd_total))

    spd_corr,spd_total = sim.runsim(truth,
                                    censor_rate,
                                    voting.knn_weighted_majority_vote,
                                    coauthor_spd,
                                    k=k)
    print('SPD: %.2f' % (spd_corr/spd_total))

    rd_corr,rd_total = sim.runsim(truth,
                                  censor_rate,
                                  voting.knn_weighted_majority_vote,
                                  coauthor_rd, 
                                  k=k)
    print('RD: %.2f' % (rd_corr/rd_total))
