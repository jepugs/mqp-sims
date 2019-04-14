import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from decimal import Decimal
from itertools import count
from pathlib import Path
import sim
import metrics
import voting

import plotting

import completeGraphs as cg
import sim_cwba as cwba


def test_cg():
    n = 250
    p = Decimal(str(0.0))
    q = Decimal(str(0.0))
    censorP = 0.7
    avgRuns = 5
    increment = Decimal(str(0.05))
    spdaccs = []
    dsdaccs = []
    rdaccs = []
    params = []

    r = 100
    
    p += increment
##    while float(p) <= 1:
##        q = Decimal(str(0.5))
##        A, truth = cg.construct_adj(n,float(p),float(q))
##        #A, truth = cg.constructWithHubs(n,float(p),float(q),r)
##        spdcorr, spdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.spd_mat(A), avgRuns)
##        dsdcorr, dsdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.dsd_mat(A), avgRuns)
##        rdcorr, rdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.rd_mat(A), avgRuns)
##        spdaccs.append(spdcorr/spdtotal)
##        dsdaccs.append(dsdcorr/dsdtotal)
##        rdaccs.append(rdcorr/rdtotal)
##        p += increment
##        params.append(float(p))
##    plotting.plot_params_vs_accuracy(params, [spdaccs, dsdaccs, rdaccs], "Edge addition probability (p)", ["SPD","DSD", "RD"], "NCC (q=0.5)")

    p = Decimal(str(0.5))
    q = Decimal(str(0.0))
    spdaccs = []
    dsdaccs = []
    rdaccs = []
    params = []

    q += increment
    while float(q) <= 1:
        A, truth = cg.construct_adj(n,float(p),float(q))
        #A, truth = cg.constructWithHubs(n,float(p),float(q),r)
        spdcorr, spdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.spd_mat(A), avgRuns)
        dsdcorr, dsdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.dsd_mat(A), avgRuns)
        rdcorr, rdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.rd_mat(A), avgRuns)
        spdaccs.append(spdcorr/spdtotal)
        dsdaccs.append(dsdcorr/dsdtotal)
        rdaccs.append(rdcorr/rdtotal)
        q += increment
        params.append(float(q))
    plotting.plot_params_vs_accuracy(params, [spdaccs, dsdaccs, rdaccs], "Edge deletion probability (q)", ["SPD","DSD", "RD"],"NCCH (p=0.5, number of hubs=100)")
    return

#test_cg()

def test_cg_h():
    n = 250
    p = Decimal(str(0.5))
    q = Decimal(str(0.5))
    censorP = 0.7
    avgRuns = 10
    increment = Decimal(str(1.0))
    spdaccs = []
    dsdaccs = []
    rdaccs = []
    params = []

    r = Decimal(str(0.0))
    
    r += increment
    while float(r) <= 20:
        #A, truth = cg.construct_adj(n,float(p),float(q))
        A, truth = cg.constructWithHubs(n,float(p),float(q),int(r))
        spdcorr, spdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.spd_mat(A), avgRuns)
        dsdcorr, dsdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.dsd_mat(A), avgRuns)
        rdcorr, rdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.rd_mat(A), avgRuns)
        spdaccs.append(spdcorr/spdtotal)
        dsdaccs.append(dsdcorr/dsdtotal)
        rdaccs.append(rdcorr/rdtotal)
        r += increment
        params.append(int(r))
    plotting.plot_params_vs_accuracy(params, [spdaccs, dsdaccs, rdaccs], "Number of hubs", ["SPD","DSD", "RD"])

    increment = Decimal(str(50.0))
    spdaccs = []
    dsdaccs = []
    rdaccs = []
    params = []

    r = Decimal(str(50.0))
    
    r += increment
    while float(r) <= 400:
        #A, truth = cg.construct_adj(n,float(p),float(q))
        A, truth = cg.constructWithHubs(n,float(p),float(q),int(r))
        spdcorr, spdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.spd_mat(A), avgRuns)
        dsdcorr, dsdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.dsd_mat(A), avgRuns)
        rdcorr, rdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.rd_mat(A), avgRuns)
        spdaccs.append(spdcorr/spdtotal)
        dsdaccs.append(dsdcorr/dsdtotal)
        rdaccs.append(rdcorr/rdtotal)
        r += increment
        params.append(int(r))
    plotting.plot_params_vs_accuracy(params, [spdaccs, dsdaccs, rdaccs], "Number of hubs", ["SPD","DSD", "RD"])
    return

#test_cg_h()

def test_cwba():
    n = 1000
    rho = Decimal(str(2.0))
    m = Decimal(str(0.0))
    censorP = 0.7
    avgRuns = 10
    increment = Decimal(str(1.0))
    spdaccs = []
    dsdaccs = []
    rdaccs = []
    params = []
    
    m += increment
    while int(m) < 20:
        #A, truth = cg.construct_adj(n,float(p),float(q))
        #A, truth = cg.constructWithHubs(n,float(p),float(q),r)
        A, truth = cwba.cwba_graph(n,int(m),float(rho))
        spdcorr, spdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.spd_mat(A), avgRuns)
        dsdcorr, dsdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.dsd_mat(A), avgRuns)
        rdcorr, rdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.rd_mat(A), avgRuns)
        spdaccs.append(spdcorr/spdtotal)
        dsdaccs.append(dsdcorr/dsdtotal)
        rdaccs.append(rdcorr/rdtotal)
        m += increment
        params.append(int(m))
    plotting.plot_params_vs_accuracy(params, [spdaccs, dsdaccs, rdaccs], "Minimum vertex degree (m)", ["SPD","DSD", "RD"], "CWBA ("+u"\u03C1"+"=2)")

#test_cwba()

def test_cwba_inv():
    n = 1000
    rho_inv = Decimal(str(0.0))
    m = Decimal(str(300.0))
    censorP = 0.7
    avgRuns = 10
    increment = Decimal(str(0.05))
    spdaccs = []
    dsdaccs = []
    rdaccs = []
    params = []
    
    rho_inv += increment
    while float(rho_inv) < 1:
        #A, truth = cg.construct_adj(n,float(p),float(q))
        #A, truth = cg.constructWithHubs(n,float(p),float(q),r)
        A, truth = cwba.cwba_graph(n,int(m),1/float(rho_inv))
        spdcorr, spdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.spd_mat(A), avgRuns)
        dsdcorr, dsdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.dsd_mat(A), avgRuns)
        rdcorr, rdtotal = sim.runsim(truth, censorP, voting.scipy_weighted_knn, metrics.rd_mat(A), avgRuns)
        spdaccs.append(spdcorr/spdtotal)
        dsdaccs.append(dsdcorr/dsdtotal)
        rdaccs.append(rdcorr/rdtotal)
        rho_inv += increment
        params.append(float(rho_inv))
    plotting.plot_params_vs_accuracy(params, [spdaccs, dsdaccs, rdaccs], "1/rho", ["SPD","DSD", "RD"])

#test_cwba_inv()

def test_coauthor():
    spdaccs, dsdaccs, rdaccs, params = [], [], [], []
    f = open('./coauthor_acc_vs_k.csv', 'r')
    for line in f.readlines():
        line = line.rstrip()
        k, spd, dsd, rd = line.split(',')
        if k == "k":
            continue
        params.append(float(k))
        spdaccs.append(float(spd))
        dsdaccs.append(float(dsd))
        rdaccs.append(float(rd))
    plotting.plot_params_vs_accuracy(params, [spdaccs, dsdaccs, rdaccs], "k nearest neighbors (k)", ["SPD","DSD", "RD"], "Coauthorship Citation Network")
    f.close()

# test_coauthor()

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
        p += Decimal(str(0.05))
        print(testacc)
    return acc, param


"""
Test case for complete graphs
"""


def test_completeGraphs(n, p, q, censorP, vote, metric, avgRuns):
    A, truth = cg.construct_adj(n, p, q)
    correct, total = sim.runsim(truth, censorP, vote, metric(A), avgRuns)
    return correct / total


def runtest_completeGraphs():
    n = 200
    q = 0.5
    censorP = 0.3
    avgRuns = 10
    dsdacc, dsdparam = suite_completeGraphs(n, q, test_completeGraphs, censorP, voting.scipy_weighted_knn, metrics.dsd_mat, avgRuns)
    spdacc, spdparam = suite_completeGraphs(n, q, test_completeGraphs, censorP, voting.scipy_weighted_knn, metrics.spd_mat, avgRuns)
    rdacc, rdparam = suite_completeGraphs(n, q, test_completeGraphs, censorP, voting.scipy_weighted_knn, metrics.rd_mat, avgRuns)
    plotting.plot_params_vs_accuracy(spdparam, [spdacc, dsdacc, rdacc], "p", ["SPD","DSD", "RD"])
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

# For 3d plots
def threeDplot():
    n = 250
    censorP = 0.7
    avgRuns = 5
    Xs,Ys,Zs = [],[],[]
    p,q = Decimal(str(0.0)),Decimal(str(0.0))
    increment = 0.05
    while p <= 1:
        while q <= 1:
            Xs.append(float(p))
            Ys.append(float(q))
            #
            G,truth = cg.construct_adj(n,p,q)
            correct,total = sim.runsim(truth,censorP,voting.scipy_weighted_knn,metrics.dsd_mat(G),avgRuns)
            acc = correct/total
            #
            Zs.append(float(acc))
            q += Decimal(str(increment))
        p += Decimal(str(increment))
        q = Decimal(str(0.0))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(Xs,Ys,Zs,cmap='Greens')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return Xs,Ys,Zs
#threeDplot()


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


####################################################################################################

### Real-world data

def test_rw_cv(truth, spd, dsd, rd, k=20, n_folds=10, verbose=False):
    '''Test a real-world data set using cross-validation. Truth should be a dict, and spd, dsd, and rd
 should be matrices.

    '''

    dsd_corr,dsd_total = sim.runsim_cv(truth, 
                                       voting.scipy_weighted_knn,
                                       dsd, 
                                       n_folds=n_folds,
                                       k=k)
    if verbose:
        print('DSD: %.2f (%d/%d)' % (dsd_corr/dsd_total, dsd_corr, dsd_total))

    spd_corr,spd_total = sim.runsim_cv(truth,
                                       voting.scipy_weighted_knn,
                                       spd,
                                       n_folds=n_folds,
                                       k=k)
    if verbose:
        print('SPD: %.2f (%d/%d)' % (spd_corr/spd_total, spd_corr, spd_total))

    rd_corr,rd_total = sim.runsim_cv(truth,
                                     voting.scipy_weighted_knn,
                                     rd, 
                                     n_folds=n_folds,
                                     k=k)
    if verbose:
        print('RD: %.2f (%d/%d)' % (rd_corr/rd_total, rd_corr, rd_total))

    return (spd_corr/spd_total, dsd_corr/dsd_total, rd_corr/rd_total)

def test_rw(truth, spd, dsd, rd, censor_rate=0.7, k=20, n_runs=10, verbose=False):
    '''Test a real-world data set using random censoring over multiple runs. Truth should be a dict, and
 spd, dsd, and rd should be matrices.

    '''

    dsd_corr,dsd_total = sim.runsim(truth,
                                    censor_rate,
                                    voting.scipy_weighted_knn,
                                    dsd, 
                                    k=k,
                                    avg_runs=n_runs)
    if verbose:
        print('DSD: %.2f (%d/%d)' % (dsd_corr/dsd_total, dsd_corr, dsd_total))

    spd_corr,spd_total = sim.runsim(truth,
                                    censor_rate,
                                    voting.scipy_weighted_knn,
                                    spd,
                                    k=k,
                                    avg_runs=n_runs)
    if verbose:
        print('SPD: %.2f (%d/%d)' % (spd_corr/spd_total, spd_corr, spd_total))

    rd_corr,rd_total = sim.runsim(truth,
                                  censor_rate,
                                  voting.scipy_weighted_knn,
                                  rd, 
                                  k=k,
                                  avg_runs=n_runs)

    if verbose:
        print('RD: %.2f (%d/%d)' % (rd_corr/rd_total, rd_corr, rd_total))

    return (spd_corr/spd_total, dsd_corr/dsd_total, rd_corr/rd_total)


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


def test_coauthor_cv(n_folds=5, k=20, verbose=False):
    truth = dict(zip(count(), np.loadtxt(coauthor_truth_filename)))
    return test_rw_cv(truth, coauthor_spd, coauthor_dsd, coauthor_rd, n_folds, k, verbose)

def test_coauthor(censor_rate=0.7, k=20, n_runs=10, verbose=False):
    truth = dict(zip(count(), np.loadtxt(coauthor_truth_filename)))
    return test_rw(truth, coauthor_spd, coauthor_dsd, coauthor_rd, censor_rate, k, n_runs, verbose)

def test_coauthor_k(k_range, censor_rate=0.7, n_runs=5, verbose=True):
    if verbose:
        print('*** Coauthor Network (censor_rate=%.2f,n_runs=%d) ***' % (censor_rate, n_runs))
        print('============================================================')
    res = []
    for k in k_range:
        x = test_coauthor(censor_rate=n_runs, n_runs=n_runs, k=k)
        if verbose:
            print('--------------------')
            print('** k = %d **' % k)
            print('SPD: %.2f\nDSD: %.2f\nRD: %.2f' % x)
        res.append(x)
    return res

# here's some gnarly copy/paste. Sorry

# emailship graph files (not included in repo)
email_truth_filename = 'email-Eu-coreLabels.txt'
email_dsd_filename = 'email-Eu-coreDSD.txt'
email_spd_filename = 'email-Eu-coreSPD.txt'
email_rd_filename = 'email-Eu-coreRD.txt'

email_truth = None
email_dsd = None
email_spd = None
email_rd = None

email_truth = np.loadtxt(email_truth_filename, delimiter=' ', dtype=np.int) if \
    fexists(email_truth_filename) and email_truth is None else email_truth
email_dsd = np.loadtxt(email_dsd_filename, delimiter=' ') if \
    fexists(email_dsd_filename) and email_dsd is None else email_dsd
email_spd = np.loadtxt(email_spd_filename, delimiter=' ') if \
    fexists(email_truth_filename) and email_spd is None else email_spd
email_rd = np.loadtxt(email_rd_filename, delimiter=' ') if \
    fexists(email_truth_filename) and email_rd is None else email_rd


def test_email_cv(n_folds=5, k=20, verbose=False):
    truth = dict(email_truth)
    return test_rw_cv(truth, email_spd, email_dsd, email_rd, n_folds, k, verbose)

def test_email(censor_rate=0.7, k=20, n_runs=10, verbose=False):
    truth = dict(email_truth)
    return test_rw(truth, email_spd, email_dsd, email_rd, censor_rate, k, n_runs, verbose)
