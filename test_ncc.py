from decimal import Decimal
import sim
import metrics
import voting
import plotting
import ncc

n = 250
p,q = Decimal(str(0.0)),Decimal(str(0.0))
censorP = 0.7
avgRuns = 10
increment = Decimal(str(0.025))

vote = voting.sklearn_weighted_knn

def test_ncc_p():
    # Initialize parameters
    global n,censorP,avgRuns,increment,vote
    p = Decimal(str(0.0))
    q = Decimal(str(0.5))
    accs_spd,accs_dsd,accs_rd = [],[],[]
    params = []

    # Run simulations
    while float(p) <= 1:
        A, truth = ncc.construct_adj(n,float(p),float(q))
        correct_spd, total_spd = sim.runsim(truth, censorP,
                                         vote, metrics.spd_mat(A), avgRuns)
        correct_dsd, total_dsd = sim.runsim(truth, censorP,
                                            vote, metrics.dsd_mat(A), avgRuns)
        correct_rd, total_rd = sim.runsim(truth, censorP,
                                          vote, metrics.rd_mat(A), avgRuns)
        accs_spd.append(correct_spd/total_spd)
        accs_dsd.append(correct_dsd/total_dsd)
        accs_rd.append(correct_rd/total_rd)
        print(p)
        p += increment
        params.append(float(p))
    plotting.plot_params_vs_accuracy(params, [accs_spd, accs_dsd, accs_rd],
                                     "Edge addition probability (p)",
                                     ["SPD","DSD","RD"],
                                     "NCC (q=0.5)")
    return

def test_ncc_q():
    # Initialize parameters
    global n,censorP,avgRuns,increment,vote
    p = Decimal(str(0.5))
    q = Decimal(str(0.0))
    accs_spd,accs_dsd,accs_rd = [],[],[]
    params = []

    # Run simulations
    while float(q) <= 1:
        A, truth = ncc.construct_adj(n,float(p),float(q))
        correct_spd, total_spd = sim.runsim(truth, censorP,
                                         vote, metrics.spd_mat(A), avgRuns)
        correct_dsd, total_dsd = sim.runsim(truth, censorP,
                                            vote, metrics.dsd_mat(A), avgRuns)
        correct_rd, total_rd = sim.runsim(truth, censorP,
                                          vote, metrics.rd_mat(A), avgRuns)
        accs_spd.append(correct_spd/total_spd)
        accs_dsd.append(correct_dsd/total_dsd)
        accs_rd.append(correct_rd/total_rd)
        print(q)
        q += increment
        params.append(float(q))
    plotting.plot_params_vs_accuracy(params, [accs_spd, accs_dsd, accs_rd],
                                     "Edge deletion probability (q)",
                                     ["SPD","DSD","RD"],
                                     "NCC (p=0.5)")
    return
