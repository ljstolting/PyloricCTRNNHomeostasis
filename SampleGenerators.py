#import sobol
import random
import numpy as np

# def neuron_genome_samp(n,solution,width):
#     '''generate quasi-random set of n points in 12D par space (no time constants) centered on the solution in question extended 
#     out by width in all directions'''
#     genelen = len(solution)
#     CTRNNsize = int(np.sqrt(1+genelen)-1)
#     dim = genelen-CTRNNsize
#     samp = sobol.sample(dimension=dim, n_points=n, skip=random.randint(0,5000)) #set between 0 and 1 in all dimensions
#     samp = (samp - .5) * 2 #center around 0 and scale back up to [-1,1]
#     samp = samp * width
#     samp = samp + solution[:-CTRNNsize]
#     sample = np.zeros((n,genelen))
#     sample[:,dim:] = solution[dim:]
#     sample[:,:dim] = samp
#     return sample

###### HP sample params ###########
taubbounds = [10,30]
tauwbounds = [30,50]
slidingwindowbounds = [1,100]

def randomHPsample(CTRNNsize, samplesize):
    genomes = np.zeros((samplesize,(CTRNNsize*2) + 3))
    for i in range(samplesize): #first, set random bias time constant, weight time constant, and sliding window length
        HPgenome = np.array([1,1,1,0,0,0,np.random.uniform(low=tauwbounds[0],high=tauwbounds[1]),np.random.uniform(low=taubbounds[0],high=taubbounds[1]),np.random.randint(low=slidingwindowbounds[0],high=slidingwindowbounds[1])])
        for j in range(CTRNNsize): #for every neuron, set a lower and upper bound
            while HPgenome[j] > .5:
                HPgenome[j] = 1-np.random.power(1)  #could have just had uniform distributuion like the rest, but I felt the need to sample the most solutions with wide target ranges
            while HPgenome[j+CTRNNsize] < .5:
                HPgenome[j+CTRNNsize] = np.random.power(1)
        genomes[i] = HPgenome
    return genomes

###### CTRNN base state sample params ########
wtbounds = [-10,10]
biasbounds = [-10,10]
taubounds = [1,2]

def randomCTRNNsample(CTRNNsize,samplesize,center_crossing=True):
    n = CTRNNsize
    genelength = (n**2)+(2*n)
    genomes = np.zeros((samplesize,genelength))
    for i in range(samplesize):
        wts = np.random.uniform(low = wtbounds[0],high = wtbounds[1],size = (n**2))
        wts2D = np.reshape(wts,(n,n))
        biases = np.random.uniform(low = biasbounds[0],high = biasbounds[1],size = n)
        taus = np.random.uniform(low = taubounds[0],high = taubounds[1],size = n)
        if center_crossing == True:
            for neuron in range(n):
                biases[neuron] = np.clip(-np.sum(wts2D[:,neuron])/2,biasbounds[0],biasbounds[1]) #set to clipped center crossing condition
        CTRNNgenome = np.concatenate((wts,biases,taus),axis=None)
        genomes[i] = CTRNNgenome
    return genomes
