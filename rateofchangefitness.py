import numpy as np
from CTRNNclass import *

N = 3
dt=.1
transient = int(100/dt) #in timesteps
testdur = transient     #in timesteps

def roc_fitness1(neurongenome,HPgenome=None,target=.5):
    '''calculate the average rate of change, compare it to the target and penalize for differences high or low'''
    if np.all(HPgenome) == None:
        #print('You have not specified an HP genome, so HP will not be used')
        HPgenome = np.ones(2*N+3)
        HPgenome[0:N] = 0
    C = CTRNN(N,dt,(transient*dt)+(testdur*dt),HPgenome,neurongenome)
    for i in range(len(C.time)):
        C.ctrnnstep(1)
    avgroc = np.sum(np.abs(np.diff(C.ctrnn_record[-testdur:])))/testdur
    fitness = 1-np.absolute(target-avgroc)
    return fitness


K = 10 #max possible fitness (arbitrary)
target = np.array([1,1,1])  # could be different for each neuron


def roc_fitness2(neurongenome,HPgenome,plotting = False):
    '''fitness = K - sumoverneurons(|T_i - C_i/D|)
    K is the maximum possible fitness (arbitrarily set a large enough number, for now 10),
    N are the number of neurons in the circuit (testing only with 3-neuron circuits for now), 
    T_i is the target rate of change of the output desired for each neuron (set to 1.0 for now), 
    C_i is the accumulated rate of change of the output for each neuron over the total duration D 
    (in units of time), calculated according to sumovertime((|o_i(t)-o_i(t-h)|)/h) 
    where o_i(t) is the output of neuron i at time t, and h is the time step of integration.
    This value is taken pre-HP and during HP-on condition and averaged'''
    HPoffgenome = np.ones(2*N+3)
    HPoffgenome[0:N] = 0
    C = CTRNN(N,dt,(transient*dt)+(testdur*dt),HPoffgenome,neurongenome) #HP off
    C.initializeOutput(np.ones(N)*.5)
    for i in range(len(C.time)):
        C.ctrnnstep(1)
    preHPfitness = K
    for n in range(N):
        acc_roc = np.sum(np.abs(np.diff(C.ctrnn_record[-testdur:])))/dt
        preHPfitness -= np.abs(target[n]-(acc_roc/testdur))
    if plotting:
        C.plot()
        print('preHPfitness=',preHPfitness)
    C = CTRNN(N,dt,(testdur*dt),HPgenome,neurongenome) #HP on, no transient
    C.initializeOutput(np.ones(N)*.5)
    for i in range(len(C.time)):
        C.ctrnnstep(1)
    HPonfitness = K
    for n in range(N):
        acc_roc = np.sum(np.abs(np.diff(C.ctrnn_record[-testdur:])))/dt
        HPonfitness -= np.abs(target[n]-(acc_roc/testdur))
    if plotting:
        C.plot()
        C.plotparams()
        print('HPonfitness=',HPonfitness)
    return np.mean((preHPfitness,HPonfitness))


