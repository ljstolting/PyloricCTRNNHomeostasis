from CTRNNclass import *

dt = .025
transientdur = 50 #seconds passed without HP
transientlen = int(transientdur/dt) #timesteps passed without HP
testdur = 50 #seconds passed where you test whether HP would be turned on or not
testlen = int(testdur/dt) #timesteps passed where you test whether HP would be turned on or not
initial_states = np.array([[10.,10.,10.],[0.,10.,10.],[10.,0.,10.],[10.,10.,0.]])

def acceptance(HPgenome,neurongenome,specificpars=np.ones(15),plot=False): #set up so that we can evolve an HP mechanism based on its acceptability at some site
    '''HP genome is numpy array of form [lb1, lb2, lb3, ub1, ub2, ub3, tauw, taub, slidingwindow]
    rho function always assumed to terminate at 1 on either side, so slope is determined by target range
    Neuron genome is numpy array of the form [weights, biases, timeconsts]
    returns boolean of whether the HP mechanism is inactive/accepting (0 if active, 1 if inactive)'''
    slidingwindowdur = HPgenome[-1]*dt
    CTRNNsize = int(np.sqrt(1+len(neurongenome))-1)
    for IC in initial_states:
        C = CTRNN(CTRNNsize,dt,transientdur+testdur+slidingwindowdur,HPgenome,neurongenome,specificpars)
        #print(C.Biases)
        C.initializeState(IC)
        C.resetStepcount()
        for i in range(transientlen+int(HPgenome[-1])):        #run the CTRNN for the transient and the length of the slidign window without checking HP 
            C.ctrnnstep(0)
        for i in range(testlen):             #run for the test period, checking HP every step
            C.ctrnnstep(0) #pars not actually changing, though
            #print('outputs=',C.Outputs)
            #print('rhos=',C.rhos)
            if C.rhos.any() != 0:
                if plot:
                    C.plotparams()
                    plt.show()
                return 0                     #if HP turns on in any of the initial conditions, it does not accept
                break
        if plot:
            C.plotparams()
            plt.show()
    return 1                                 #if it never turns on for any of the IC's, it accepts