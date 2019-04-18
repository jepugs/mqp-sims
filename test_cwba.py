from decimal import Decimal
import sim
import metrics
import voting
import plotting
import cwba

n = 1000
rho = Decimal(str(2.0))
m = Decimal(str(0.0))
censorP = 0.7
avgRuns = 10
increment = Decimal(str(0.025))

vote = voting.sklearn_weighted_knn

def test_cwba_m():
    # Initialize parameters
    global n,censorP,avgRuns,vote
    rho = Decimal(str(2.0))
    m = Decimal(str(1.0))
    increment = Decimal(str(1.0))
    accs_spd,accs_dsd,accs_rd = [],[],[]
    params = []

    # Run simulations
    while int(m) < 20:
        A, truth = cwba.cwba_graph(n,int(m),float(rho))
        correct_spd, total_spd = sim.runsim(truth, censorP,
                                         vote, metrics.spd_mat(A), avgRuns)
        correct_dsd, total_dsd = sim.runsim(truth, censorP,
                                            vote, metrics.dsd_mat(A), avgRuns)
        correct_rd, total_rd = sim.runsim(truth, censorP,
                                          vote, metrics.rd_mat(A), avgRuns)
        accs_spd.append(correct_spd/total_spd)
        accs_dsd.append(correct_dsd/total_dsd)
        accs_rd.append(correct_rd/total_rd)
        print(m)
        params.append(int(m))
        m += increment
    rho_str = u"\u03C1"
    plotting.plot_params_vs_accuracy(params, [accs_spd, accs_dsd, accs_rd],
                                     "New vertex degree (m)",
                                     ["SPD","DSD","RD"],
                                     "CWBA ("+rho_str+"=2)")
    return

def test_cwba_rhoinv():
    # Initialize parameters
    global n,censorP,avgRuns,vote
    rho_inv = Decimal(str(0.05))
    m = Decimal(str(300.0))
    increment = Decimal(str(0.05))
    accs_spd,accs_dsd,accs_rd = [],[],[]
    params = []

    # Run simulations
    while float(rho_inv) <= 1:
        A, truth = cwba.cwba_graph(n,int(m),1/float(rho_inv))
        correct_spd, total_spd = sim.runsim(truth, censorP,
                                         vote, metrics.spd_mat(A), avgRuns)
        correct_dsd, total_dsd = sim.runsim(truth, censorP,
                                            vote, metrics.dsd_mat(A), avgRuns)
        correct_rd, total_rd = sim.runsim(truth, censorP,
                                          vote, metrics.rd_mat(A), avgRuns)
        accs_spd.append(correct_spd/total_spd)
        accs_dsd.append(correct_dsd/total_dsd)
        accs_rd.append(correct_rd/total_rd)
        print(rho_inv)
        params.append(float(rho_inv))
        rho_inv += increment
    rho_str = u"\u03C1"
    plotting.plot_params_vs_accuracy(params, [accs_spd, accs_dsd, accs_rd],
                                     "Inverse likeliness of clusters (1/"+rho_str+")",
                                     ["SPD","DSD","RD"],
                                     "CWBA (m=300)")
    return
