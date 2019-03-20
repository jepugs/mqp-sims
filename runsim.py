import networkx as nx
import numpy as np
import datetime

import sim_complete_hubs as comp_hubs
import sim_writeGraph as writeG
import sim_vote as vote
import sim_censor as censor
import runDSD as dsd
import sim_plot as splot
import sim_wsg as wsg
import mydsd as mydsd
import sim_regular_hubs as reg_hubs

# Run simulations

'''
    Complete with Hubs simulation
'''
def runCompleteSim():
    # Number of nodes in each complete graph
    n = 50
    m = 50
    # Number of hub vertices
    hubs = 0
    # Probability adding an edge to a hub node
    hp = 0.2
    # Probability of flipping an edge
    p = 0.0

    accuracy = []
    pdata = []
    while p <= 1:
        print('probability: ',p)
        pdata.append(p)
        avgacc = 0
        for avgi in range(10):
            G = comp_hubs.simulate(n,m,hubs,hp,p)
            i = 0
            G_censored = censor.censorGraph(G.copy(), 0.3)
            writeG.writeGraphFile('test'+str(i), G)
            DSD, names = dsd.runDSD('./DSD/testin/test_test'+str(i)+'.csv', './DSD/testout/sim_test'+str(i))
            rnames = []
            for ni,name in enumerate(list(names)):
                rnames.append(name[1:len(name)-1])
            #DSD = mydsd.dsdMat(G_censored)
            #rnames = list(G.nodes())

            #splot.plotDistanceFrequency(DSD, 'DSD')

            #pred,t = vote.majorityVote(G_censored,4,DSD,G,rnames)
            pred,t = vote.majorityWeightedVote(G_censored,10,DSD,G,rnames)
            avgacc += pred/t
        print(avgacc/10)
        accuracy.append(avgacc/10)
        p += 0.05
    # Plot data
    splot.plotAccuracyToParam(accuracy, pdata, 'p')

def runCompleteSimShortestPaths():
    # Number of nodes in each complete graph
    n = 50
    m = 50
    # Number of hub vertices
    hubs = 0
    # Probability adding an edge to a hub node
    hp = 0.2
    # Probability of flipping an edge
    p = 0.0

    accuracy = []
    pdata = []
    while p <= 1:
        print('probability: ',p)
        pdata.append(p)
        avgacc = 0
        for avgi in range(10):
            G = comp_hubs.simulate(n,m,hubs,hp,p)
            i = 0
            G_censored = censor.censorGraph(G.copy(), 0.3)
            writeG.writeGraphFile('test'+str(i), G)
            SP = np.zeros((len(G), len(G)))
            rnames = []
            for i, vertexi in enumerate(G):
                for j, vertexj in enumerate(G):
                    if i != j:
                        SP[i,j] = nx.shortest_path_length(G,vertexi,vertexj)

            rnames = [v for v in G.nodes()]

            #pred,t = vote.majorityVote(G_censored,4,DSD,G,rnames)
            pred,t = vote.majorityWeightedVote(G_censored,4,SP,G,rnames)
            avgacc += pred/t
        print(avgacc/10)
        accuracy.append(avgacc/10)
        p += 0.05
    # Plot data
    splot.plotAccuracyToParam(accuracy, pdata, 'p')

def runWSG():
    n = 200
    k = 10
    p = 0.1
    q = 0
    hubs = 5
    hubsp = 0.4

    accuracy = []
    pdata = []
    while q <= 1:
        print('probability: ',q)
        pdata.append(q)
        avgacc = 0
        for avgi in range(10):
            G = wsg.simulate(n,k,p,q,hubs,hubsp)
            i = 0
            G_censored = censor.censorGraph(G.copy(), 0.4)
            writeG.writeGraphFile('test'+str(i), G)
            DSD, names = dsd.runDSD('./DSD/testin/test_test'+str(i)+'.csv', './DSD/testout/sim_test'+str(i))
            #DSD = mydsd.dsdMat(G)
            #rnames = list(G.nodes())
            #splot.plotDistanceFrequency(DSD, 'DSD')

            rnames = []
            for ni,name in enumerate(list(names)):
                rnames.append(name[1:len(name)-1])

            #pred,t = vote.majorityVote(G_censored,4,DSD,G,rnames)
            pred,t = vote.majorityWeightedVote(G_censored,4,DSD,G,rnames)
            avgacc += pred/t
        print(avgacc/10)
        accuracy.append(avgacc/10)
        q += 0.05
    # Plot data
    splot.plotAccuracyToParam(accuracy, pdata, 'q')

def runWSGShortestPaths():
    n = 200
    k = 10
    p = 0.0
    q = 0.0
    hubs = 5
    hubsp = 0.4

    accuracy = []
    pdata = []
    while q <= 1:
        print('probability: ',q)
        pdata.append(q)
        avgacc = 0
        for avgi in range(10):
            G = wsg.simulate(n,k,p,q,hubs,hubsp)
            print('size: ',len(G))
            i = 0
            G_censored = censor.censorGraph(G.copy(), 0.4)
            writeG.writeGraphFile('test'+str(i), G)
            SP = np.zeros((len(G), len(G)))
            rnames = []
            for i, vertexi in enumerate(G):
                for j, vertexj in enumerate(G):
                    if i != j:
                        SP[i,j] = nx.shortest_path_length(G,vertexi,vertexj)

            rnames = [v for v in G.nodes()]

            #pred,t = vote.majorityVote(G_censored,4,DSD,G,rnames)
            pred,t = vote.majorityWeightedVote(G_censored,10,SP,G,rnames)
            avgacc += pred/t
        print(avgacc/10)
        accuracy.append(avgacc/10)
        q += 0.05
    # Plot data
    splot.plotAccuracyToParam(accuracy, pdata, 'q')


def runRegularSim():
    # Number of nodes in each d-regular graph
    n = 200
    d = 20
    # Number of hub vertices
    hubs = 10
    # Probability adding an edge to a hub node
    hp = 0.8
    # Probability of adding an edge between the regular graphs
    p = 0.0

    accuracy = []
    pdata = []
    while p <= 1:
        print('probability: ',p)
        pdata.append(p)
        avgacc = 0
        for avgi in range(10):
            G = reg_hubs.simulate(n,d,p,hubs,hp)
            i = 0
            G_censored = censor.censorGraph(G.copy(), 0.4)
            writeG.writeGraphFile('test'+str(i), G)
            DSD, names = dsd.runDSD('./DSD/testin/test_test'+str(i)+'.csv', './DSD/testout/sim_test'+str(i))
            #splot.plotDistanceFrequency(DSD, 'DSD')

            rnames = []
            for ni,name in enumerate(list(names)):
                rnames.append(name[1:len(name)-1])

            #pred,t = vote.majorityVote(G_censored,4,DSD,G,rnames)
            pred,t = vote.majorityWeightedVote(G_censored,10,DSD,G,rnames)
            avgacc += pred/t
        print(avgacc/10)
        accuracy.append(avgacc/10)
        p += 0.025
    # Plot data
    splot.plotAccuracyToParam(accuracy, pdata, 'p')

def runRegularSimShortestPaths():
    # Number of nodes in each d-regular graph
    n = 200
    d = 20
    # Number of hub vertices
    hubs = 10
    # Probability adding an edge to a hub node
    hp = 0.8
    # Probability of adding an edge between the regular graphs
    p = float(0.0)

    accuracy = []
    pdata = []
    while p <= 1:
        print('probability: ',p)
        pdata.append(p)
        avgacc = 0
        for avgi in range(10):
            G = reg_hubs.simulate(n,d,p,hubs,hp)
            i = 0
            G_censored = censor.censorGraph(G.copy(), 0.4)
            writeG.writeGraphFile('test'+str(i), G)
            SP = np.zeros((len(G), len(G)))
            rnames = []
            for i, vertexi in enumerate(G):
                for j, vertexj in enumerate(G):
                    if i != j:
                        SP[i,j] = nx.shortest_path_length(G,vertexi,vertexj)

            rnames = [v for v in G.nodes()]

            #pred,t = vote.majorityVote(G_censored,4,DSD,G,rnames)
            pred,t = vote.majorityWeightedVote(G_censored,4,SP,G,rnames)
            avgacc += pred/t
        print(avgacc/10)
        accuracy.append(avgacc/10)
        p += float(0.025)
    # Plot data
    splot.plotAccuracyToParam(accuracy, pdata, 'p')


#runRegularSim()
#runRegularSimShortestPaths()

runCompleteSim()
runCompleteSimShortestPaths()

#runWSG()
#runWSGShortestPaths()

##G = wsg.simulate(200,10,0.1,0,0,0.4)
##i = 0
##writeG.writeGraphFile('test'+str(i), G)
##DSD, names = dsd.runDSD('./DSD/testin/test_test'+str(i)+'.csv', './DSD/testout/sim_test'+str(i))
##DSD2 = mydsd.dsdMat(G)
