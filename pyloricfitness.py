import numpy as np

from CTRNNclass import *

#####To do: make doubly periodic solutions not count##########

burst_on_thresh = .5 #the "firing rate threshold" at which the CTRNN neuron is considered to be "bursting"
burst_off_thresh = .45  #the "firing rate" at which the neuron is considered to be effectively silent (just used to determine if oscillation is wide enough)

initial_states = np.array([3.,3.,3.])  #initial states of the neurons
dt=.01
transientdur = 100 #in seconds
transient = int(transientdur/dt) #in timesteps
duration = 250 #time to simulate CTRNN for in seconds

def pyloriclike(neurongenome,HPgenome = None,specificpars=np.ones(15),debugging=False):
    '''input is CTRNN genome [weights,biases,timeconsts] and HP genome is [lbs,ubs,taub,tauw,slidingwindow]. 
    Output is its fitness as pyloric-like rhythm. Awards .05 for each oscillating neuron, and .05 for each 
    order critereon met. Then, if all order criteria met, adds (1/z-score) for each of 15 criteria in table 1'''
    CTRNNsize = int(np.sqrt(1+len(neurongenome))-1)
    if np.all(HPgenome) == None:
        #print('You have not specified an HP genome, so HP will not be used')
        HPgenome = np.ones(2*CTRNNsize+3)
        HPgenome[0:CTRNNsize] = 0
        HP = 0
    else:
        HP = 1
    C = CTRNN(CTRNNsize,dt,duration,HPgenome,neurongenome,specificpars)
    C.initializeState(initial_states)
    C.resetStepcount()
    for i in range(len(C.time)):        #run the CTRNN for the allotted duration
        C.ctrnnstep(HP)
    # C.plot()
    #check if first three neurons were oscillating (all the way from silent to burst) by the end of the run
    osc = np.zeros(3)
    for i in range(3):
        #print(max(C.ctrnn_record[i,transient:]),min(C.ctrnn_record[i,transient:]))
        if max(C.ctrnn_record[i,transient:]) > burst_on_thresh:
            if min(C.ctrnn_record[i,transient:]) < burst_on_thresh-.025:
                osc[i] = 1
    #print(osc)
    fitness = sum(osc)*0.05 #initialize a fitness value based on how many neurons oscillate sufficiently
    if np.all(osc):
        #LP = N1, PY = N2, PD = N3
        #Scan to find second to last full PD cycle
        PDstart3 = 0
        PDstart2 = 0
        PDstart1 = 0
        for i in range(len(C.time))[:transient:-1]:
            if C.ctrnn_record[2,i] > burst_on_thresh:
                if C.ctrnn_record[2,i-1] < burst_on_thresh:
                    PDstart3 = i
                    break
        for i in range(PDstart3)[:transient:-1]:
            if C.ctrnn_record[2,i] > burst_on_thresh:
                if C.ctrnn_record[2,i-1] < burst_on_thresh:
                    PDstart2 = i
                    break
        for i in range(PDstart2)[:transient:-1]:
            if C.ctrnn_record[2,i] > burst_on_thresh:
                if C.ctrnn_record[2,i-1] < burst_on_thresh:
                    PDstart1 = i
                    break
        if (PDstart1 == 0 or PDstart2 == 0):
            if debugging == True:
                print('unable to find two full cycles,may want to increase runtime')
                print('CTRNN',neurongenome)
                #print('HP',HPgenome)
            return fitness
        #calculate the start and end times of each neuron's burst in the last full cycle
        PDend = 0 #end of PD burst
        for i in range(PDstart1,PDstart2): 
            if C.ctrnn_record[2,i] > burst_on_thresh:
                if C.ctrnn_record[2,i+1] < burst_on_thresh:
                    PDend = i
                    break
        LPstart = []
        LPend = 0
        for i in range(PDstart1,PDstart2-1):
            if C.ctrnn_record[0,i] < burst_on_thresh:
                if C.ctrnn_record[0,i+1] > burst_on_thresh:
                    LPstart.append(i)
        if len(LPstart)!=1:
            if debugging == True:
                print('possible double-periodicity')
                print('CTRNN',neurongenome)
                #print('HP',HPgenome)
            return fitness
        LPstart = LPstart[0]
        for i in range(LPstart,len(C.time)-1):
            if C.ctrnn_record[0,i] > burst_on_thresh:
                if C.ctrnn_record[0,i+1] < burst_on_thresh:
                    LPend = i
                    break
        PYstart = []
        PYend = 0
        for i in range(PDstart1,PDstart2-1):
            if C.ctrnn_record[1,i] < burst_on_thresh:
                if C.ctrnn_record[1,i+1] > burst_on_thresh:
                    PYstart.append(i)
        if len(PYstart)!=1:
            if debugging == True:
                print('possible double-periodicity')
                print('CTRNN',neurongenome)
                #print('HP',HPgenome)
            return fitness
        PYstart = PYstart[0]
        for i in range(PYstart,len(C.time)-1):
            if C.ctrnn_record[1,i] > burst_on_thresh:
                if C.ctrnn_record[1,i+1] < burst_on_thresh:
                    PYend = i
                    break
        #print(PDstart1,PDend,PYstart,PYend,LPstart,LPend)
        #test if in right order
        if LPstart <= PYstart:
            fitness += 0.05
            #print('check1')
        if LPend <= PYend:
            fitness += 0.05
            #print('check2')
        if PDend <= LPstart:
            fitness += 0.05
            #print('check3')
        if debugging==True:
            print("LP",LPstart,",",LPend," PY",PYstart,",",PYend," PD",PDstart1,',',PDend)
        #if all oscillating in the right order, award fitness for the duty cycle criteria being close to the mean observed in crabs
        if fitness == 0.3:
            period = PDstart2 - PDstart1
            LPdutycycle = (LPend-LPstart)/period #burstduration/period
            LPdutycyclezscore = abs(LPdutycycle - .264)/.059
            PYdutycycle = (PYend-PYstart)/period #burstduration/period
            PYdutycyclezscore = abs(PYdutycycle - .348)/.054
            PDdutycycle = (PDend-PDstart1)/period #burstduration/period
            PDdutycyclezscore = abs(PDdutycycle - .385)/.040
            LPstartphase = (LPstart-PDstart1)/period #delay/period
            LPstartphasezscore = abs(LPstartphase - .533)/.054
            PYstartphase = (PYstart-PDstart1)/period #delay/period
            PYstartphasezscore = abs(PYstartphase - .758)/.060
            if debugging == True:
                print('LPdutycyclezscore ',LPdutycyclezscore)
                print('PYdutycyclezscore ',PYdutycyclezscore)
                print('PDdutycyclezscore ',PDdutycyclezscore)
                print('LPstartphasezscore ',LPstartphasezscore)
                print('PYstartphasezscore ',PYstartphasezscore)
            fitness += 1/(np.average([LPdutycyclezscore,PYdutycyclezscore,PDdutycyclezscore,LPstartphasezscore,PYstartphasezscore]))
    return fitness

HPongenome = [.25,.25,.25,.75,.75,.75,40,20,1]
def pyloriclikewithHP(neurongenome):
    return(pyloriclike(neurongenome,HPgenome=HPongenome))

def pyloricfitness(neurongenome,HPgenome = None,specificpars=np.ones(15),debugging=False):
    '''New, continuous pyloric fitness function without discrete requirements. Simply does not award extra 
    fitness for oscillation and ordering criteria. Does not check for ordering criteria. If all z-scores were
    met, theoretically ordering criteria would be met, also.'''
    CTRNNsize = int(np.sqrt(1+len(neurongenome))-1)
    if np.all(HPgenome) == None:
        HPgenome = np.ones(2*CTRNNsize+3)
        HPgenome[0:CTRNNsize] = 0
        HP = 0
    else:
        HP = 1
    C = CTRNN(CTRNNsize,dt,duration,HPgenome,neurongenome,specificpars)
    C.initializeState(initial_states)
    C.resetStepcount()
    for i in range(len(C.time)):
        C.ctrnnstep(HP)
    fitness = 0.0
    #check if all neurons were oscillating around the bursting threshold
    osc = np.zeros(3)
    for i in range(3):
        print(max(C.ctrnn_record[i,transient:]),min(C.ctrnn_record[i,transient:]))
        if max(C.ctrnn_record[i,transient:]) > burst_on_thresh+.025:
            if min(C.ctrnn_record[i,transient:]) < burst_on_thresh-.025:
                osc[i] = 1
    if np.all(osc):
        #LP = N1, PY = N2, PD = N3
        #Scan to find second to last full PD cycle
        PDstart3 = 0
        PDstart2 = 0
        PDstart1 = 0
        for i in range(len(C.time))[:transient:-1]:
            if C.ctrnn_record[2,i] > burst_on_thresh:
                if C.ctrnn_record[2,i-1] < burst_on_thresh:
                    PDstart3 = i
                    break
        for i in range(PDstart3)[:transient:-1]:
            if C.ctrnn_record[2,i] > burst_on_thresh:
                if C.ctrnn_record[2,i-1] < burst_on_thresh:
                    PDstart2 = i
                    break
        for i in range(PDstart2)[:transient:-1]:
            if C.ctrnn_record[2,i] > burst_on_thresh:
                if C.ctrnn_record[2,i-1] < burst_on_thresh:
                    PDstart1 = i
                    break
        if (PDstart1 == 0 or PDstart2 == 0):
            if debugging == True:
                print('unable to find two full cycles,may want to increase runtime')
                print('CTRNN',neurongenome)
                #print('HP',HPgenome)
            return fitness
        #calculate the start and end times of each neuron's burst in the last full cycle
        PDend = 0 #end of PD burst
        for i in range(PDstart1,PDstart2): 
            if C.ctrnn_record[2,i] > burst_on_thresh:
                if C.ctrnn_record[2,i+1] < burst_on_thresh:
                    PDend = i
                    break
        LPstart = []
        LPend = 0
        for i in range(PDstart1,PDstart2-1):
            if C.ctrnn_record[0,i] < burst_on_thresh:
                if C.ctrnn_record[0,i+1] > burst_on_thresh:
                    LPstart.append(i)
        if len(LPstart)!=1:
            if debugging == True:
                print('possible double-periodicity')
                print('CTRNN',neurongenome)
                #print('HP',HPgenome)
            return fitness
        LPstart = LPstart[0]
        for i in range(LPstart,len(C.time)-1):
            if C.ctrnn_record[0,i] > burst_on_thresh:
                if C.ctrnn_record[0,i+1] < burst_on_thresh:
                    LPend = i
                    break
        PYstart = []
        PYend = 0
        for i in range(PDstart1,PDstart2-1):
            if C.ctrnn_record[1,i] < burst_on_thresh:
                if C.ctrnn_record[1,i+1] > burst_on_thresh:
                    PYstart.append(i)
        if len(PYstart)!=1:
            if debugging == True:
                print('possible double-periodicity')
                print('CTRNN',neurongenome)
                #print('HP',HPgenome)
            return fitness
        PYstart = PYstart[0]
        for i in range(PYstart,len(C.time)-1):
            if C.ctrnn_record[1,i] > burst_on_thresh:
                if C.ctrnn_record[1,i+1] < burst_on_thresh:
                    PYend = i
                    break
        period = PDstart2 - PDstart1
        LPdutycycle = (LPend-LPstart)/period #burstduration/period
        LPdutycyclezscore = abs(LPdutycycle - .264)/.059
        PYdutycycle = (PYend-PYstart)/period #burstduration/period
        PYdutycyclezscore = abs(PYdutycycle - .348)/.054
        PDdutycycle = (PDend-PDstart1)/period #burstduration/period
        PDdutycyclezscore = abs(PDdutycycle - .385)/.040
        LPstartphase = (LPstart-PDstart1)/period #delay/period
        LPstartphasezscore = abs(LPstartphase - .533)/.054
        PYstartphase = (PYstart-PDstart1)/period #delay/period
        PYstartphasezscore = abs(PYstartphase - .758)/.060
        if debugging == True:
            print('LPdutycyclezscore ',LPdutycyclezscore)
            print('PYdutycyclezscore ',PYdutycyclezscore)
            print('PDdutycyclezscore ',PDdutycyclezscore)
            print('LPstartphasezscore ',LPstartphasezscore)
            print('PYstartphasezscore ',PYstartphasezscore)
        fitness += 1/(np.average([LPdutycyclezscore,PYdutycyclezscore,PDdutycyclezscore,LPstartphasezscore,PYstartphasezscore]))
    return fitness