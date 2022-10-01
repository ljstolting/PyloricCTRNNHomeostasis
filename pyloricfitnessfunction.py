######Earlier Version####

import numpy as np

burst_on_thresh = .5 #the "firing rate threshold" at which the CTRNN neuron is considered to be "bursting"
burst_off_thresh = .45  #the "firing rate" at which the neuron is considered to be effectively silent (just used to determine if oscillation is wide enough)

initial_states = np.array([3.,3.,3.])  #initial states of the neurons
dt=.025
transient = 5000 #in timesteps
HP = 0 #true or false for whether to apply HP

def pyloriclike(genome,HP):
    '''input is an object of the adaptive CTRNN class and output is its fitness as a pyloric-like rhythm. Awards .05 for each oscillating
    neuron, and .05 for each order critereon met. Then, if all order criteria met, adds (1/z-score) for each of 15 criteria in table 1'''
    C = CTRNN(3,dt,duration,1,offset,np.reshape(genome[0:9],(3,3)),genome[9:12],genome[12:])
    C.initializeState(initial_states)
    C.resetStepcount()
    for i in range(len(C.time)):        #run the CTRNN for the allotted duration
        C.ctrnnstep(HP)
    #check if all three neurons were oscillating (all the way from silent to burst) by the end of the run
    osc = np.zeros(C.Size)
    for i in range(C.Size):
        if max(C.ctrnn_record[i,transient:]) > burst_on_thresh:
            if min(C.ctrnn_record[i,transient:]) < burst_off_thresh:
                osc[i] = 1
    #print(osc)
    fitness = sum(osc)*0.05 #initialize a fitness value based on how many neurons oscillate sufficiently
    if np.all(osc):
        #print("we got oneeeeeee")
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
            print(genome)
            print('could not find 2 full cycles')
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
            print(genome)
            print('possible double-periodicity')
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
            print(genome)
            print('possible double-periodicity')
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
        #if all oscillating in the right order, award fitness for the 15 criteria being close to the mean
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
            fitness += 1/(np.average([LPdutycyclezscore,PYdutycyclezscore,PDdutycyclezscore,LPstartphasezscore,PYstartphasezscore]))
    return fitness       
#need to detect if more than one or no full burst from neurons 1 and 2 during the course of one cycle of neuron 3