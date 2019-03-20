import networkx as nx
import matplotlib.pyplot as plt
from decimal import Decimal
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
        p += Decimal(str(0.025))
        print(testacc)
    return acc, param


"""
Test case for complete graphs
"""


def test_completeGraphs(n, p, q, censorP, vote, metric, avgRuns):
    G, truth = cg.construct(n, p, q)
    correct, total = sim.runsim(G, truth, censorP, vote, metric(G), avgRuns)
    return correct / total


def runtest_completeGraphs():
    n = 200
    q = 0.5
    censorP = 0.3
    avgRuns = 100
    dsdacc, dsdparam = suite_completeGraphs(n, q, test_completeGraphs, censorP, voting.weightedMajorityVote, metrics.dsdMat, avgRuns)
    spdacc, spdparam = suite_completeGraphs(n, q, test_completeGraphs, censorP, voting.weightedMajorityVote, metrics.spdMat, avgRuns)
    plotting.plotAccuraciesToParam("p", dsdparam, [dsdacc, spdacc])
    return


runtest_completeGraphs()

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
