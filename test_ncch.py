from decimal import Decimal
import sim
import metrics
import voting
import plotting
import ncch

n = 250
p,q = Decimal(str(0.0)),Decimal(str(0.0))
r = 100
rp = 0.8
censorP = 0.7
avgRuns = 10
increment = Decimal(str(0.025))

vote = voting.sklearn_weighted_knn

def test_ncch_p():
    # Initialize parameters
    global n,r,rp,censorP,avgRuns,increment,vote
    p = Decimal(str(0.0))
    q = Decimal(str(0.5))
    accs_spd,accs_dsd,accs_rd = [],[],[]
    params = []

    # Run simulations
    while float(p) <= 1:
        A, truth = ncch.construct_adj(n,float(p),float(q),r,rp)
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
                                     "NCCH (q=0.5)")
    return

def test_ncch_q():
    # Initialize parameters
    global n,r,rp,censorP,avgRuns,increment,vote
    p = Decimal(str(0.5))
    q = Decimal(str(0.0))
    accs_spd,accs_dsd,accs_rd = [],[],[]
    params = []

    # Run simulations
    while float(q) <= 1:
        A, truth = ncch.construct_adj(n,float(p),float(q),r,rp)
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
                                     "NCCH (p=0.5)")
    return

def test_ncch_r():
    # Initialize parameters
    global n,rp,censorP,avgRuns,vote
    p = Decimal(str(0.2)) # p=0.2,0.8
    q = Decimal(str(0.5)) # q=0.5
    r = 100 # r=(0,20,1), r=(100,400,50)
    increment = 50
    accs_spd,accs_dsd,accs_rd = [],[],[]
    params = []

    # Run simulations
    while r <= 400:
        A, truth = ncch.construct_adj(n,float(p),float(q),r,rp)
        correct_spd, total_spd = sim.runsim(truth, censorP,
                                         vote, metrics.spd_mat(A), avgRuns)
        correct_dsd, total_dsd = sim.runsim(truth, censorP,
                                            vote, metrics.dsd_mat(A), avgRuns)
        correct_rd, total_rd = sim.runsim(truth, censorP,
                                          vote, metrics.rd_mat(A), avgRuns)
        accs_spd.append(correct_spd/total_spd)
        accs_dsd.append(correct_dsd/total_dsd)
        accs_rd.append(correct_rd/total_rd)
        print(r)
        r += increment
        params.append(r)
    plotting.plot_params_vs_accuracy(params, [accs_spd, accs_dsd, accs_rd],
                                     "Number of hubs",
                                     ["SPD","DSD","RD"],
                                     "NCCH (p="+str(p)+",q="+str(q)+")")
    return

def test_ncch_censor():
    # Initialize parameters
    global n,r,rp,avgRuns,vote
    p = Decimal(str(0.8)) # p=0.8
    q = Decimal(str(0.5)) # q=0.5
    censorP = Decimal(str(0.1))
    increment = Decimal(str(0.1))
    accs_spd,accs_dsd,accs_rd = [],[],[]
    params = []

    # Run simulations
    while float(censorP) < 1:
        A, truth = ncch.construct_adj(n,float(p),float(q),r,rp)
        correct_spd, total_spd = sim.runsim(truth, float(censorP),
                                         vote, metrics.spd_mat(A), avgRuns)
        correct_dsd, total_dsd = sim.runsim(truth, float(censorP),
                                            vote, metrics.dsd_mat(A), avgRuns)
        correct_rd, total_rd = sim.runsim(truth, float(censorP),
                                          vote, metrics.rd_mat(A), avgRuns)
        accs_spd.append(correct_spd/total_spd)
        accs_dsd.append(correct_dsd/total_dsd)
        accs_rd.append(correct_rd/total_rd)
        print(censorP)
        censorP += increment
        params.append(float(censorP))
    plotting.plot_params_vs_accuracy(params, [accs_spd, accs_dsd, accs_rd],
                                     "Vertex label censor proportion",
                                     ["SPD","DSD","RD"],
                                     "NCCH (p="+str(p)+",q="+str(q)+")")
    return
