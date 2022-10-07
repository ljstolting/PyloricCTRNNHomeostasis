import numpy as np
from CTRNNclass import *

n = 3
initial_states = np.array([3.,3.,3.])  #initial states of the neurons
dt=.025
transient = 8000 #in timesteps
testdur = 50     #in timesteps

def roc_fitness(neurongenome,HPgenome=None,target=.5):
    '''calculate the average rate of change, compare it to the target and penalize for differences high or low'''
    if np.all(HPgenome) == None:
        #print('You have not specified an HP genome, so HP will not be used')
        HPgenome = np.ones(2*n+3)
        HPgenome[0:n] = 0
    C = CTRNN(n,dt,(transient*dt)+(testdur*dt),HPgenome,neurongenome)
    for i in range(len(C.time)):
        C.ctrnnstep(1)
    avgroc = np.sum(np.diff(C.ctrnn_record[-testdur:]))/testdur
    fitness = 1-np.absolute(target-avgroc)
    return fitness
