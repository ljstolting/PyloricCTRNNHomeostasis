#the evoltuion is now no longer general (keeps lbs and ubs in order & keeps the sliding window an integer)

import numpy as np
import matplotlib.pyplot as plt

def differentialapplication(wholegenome,boolean):
    '''turn off the application of HP in a specific genome. boolean is list of ones and zeros for whether 
    to apply HP or not (0->off)'''
    genome = wholegenome
    n = int(np.sqrt(1+len(genome))-2)
    for i in range(n):
        if not(boolean[i]):
            genome[(n**2)+(2*n)+i] = 0             #nullify lower bound
            genome[(n**2)+(2*n)+i+n] = 1   #nullify upper bound
    #print(genome)
    return genome

#split into evolving HP, evolving CTRNN, and coevolving both
class MicrobialHPCTRNN():
    '''coevolving HP & CTRNN, specific to genomes of the form [neurongenome,HPgenome]'''
    def __init__(self, fitnessFunction, startpopulation, recombProb, mutatProb, generations, differentialapp):
        self.fitnessFunction = fitnessFunction
        self.popsize = len(startpopulation)
        self.genesize = len(startpopulation[0])
        self.recombProb = recombProb
        self.mutatProb = mutatProb
        self.generations = generations
        self.tournaments = generations*self.popsize
        self.pop = startpopulation
        self.CTRNNsize = int(np.sqrt(1+self.genesize)-2)
        self.fitness = np.zeros(self.popsize)
        self.avgHistory = np.zeros(generations)
        self.bestHistory = np.zeros(generations)
        self.gen = 0
        self.diffapp = differentialapp #list of booleans
    def showFitness(self):
        plt.plot(self.bestHistory)
        plt.plot(self.avgHistory)
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Best and average fitness")
        plt.show()

    def fitStats(self):
        bestind = self.pop[np.argmax(self.fitness)]
        bestfit = np.max(self.fitness)
        avgfit = np.mean(self.fitness)
        self.avgHistory[self.gen]=avgfit
        self.bestHistory[self.gen]=bestfit
        return avgfit, bestfit, bestind

    def save(self,filename):
        af,bf,bi = self.fitStats()
        np.savez(filename, avghist=self.avgHistory, besthist=self.bestHistory, bestind=bi)

    def run(self):
        # Calculate all fitness once
        for i in range(self.popsize):
            self.fitness[i] = self.fitnessFunction(self.pop[i])
        # Evolutionary loop
        for g in range(self.generations):
            #print(self.pop)
            self.gen = g
            # Report statistics every generation
            self.fitStats()
            for i in range(self.popsize):
                # Step 1: Pick 2 individuals
                a = np.random.randint(0,self.popsize-1)
                b = np.random.randint(0,self.popsize-1)
                while (a==b):   # Make sure they are two different individuals
                    b = np.random.randint(0,self.popsize-1)
                # Step 2: Compare their fitness
                if (self.fitness[a] > self.fitness[b]):
                    winner = a
                    loser = b
                else:
                    winner = b
                    loser = a
                # Step 3: Transfect loser with winner --- Could be made more efficient using Numpy
                for l in range(self.genesize):
                    if (np.random.random() < self.recombProb):
                        self.pop[loser][l] = self.pop[winner][l]
                # Step 4: Mutate loser and make sure new organism stays within bounds
                self.pop[loser] += np.concatenate((np.random.normal(0.0,self.mutatProb,size=self.genesize-1),np.random.randint(-50,50)),axis=None) #sliding windows do not mutate continuously
                if self.pop[loser,-1] < 1:
                    self.pop[loser,-1] = 1
                elif self.pop[loser,-1] > 300:
                    self.pop[loser,-1] = 300
                self.pop[loser] = differentialapplication(self.pop[loser],self.diffapp)
                for n in range(self.CTRNNsize):
                    loserlb = self.pop[loser,-(3+(2*self.CTRNNsize)-n)]
                    loserub = self.pop[loser,-(3+self.CTRNNsize-i)]
                    if loserlb >= loserub: #if lb greater than upper bound for any neuron,
                        self.pop[loser,-(3+(2*self.CTRNNsize)-n)] = loserub - .01 #minimum width of range is .01
                    if loserlb < 0:
                        self.pop[loser,-(3+(2*self.CTRNNsize)-n)] = 0
                    if loserub > 1:
                        self.pop[loser,-(3+self.CTRNNsize-i)] = 1
                # Save fitness
                self.fitness[loser] = self.fitnessFunction(self.pop[loser])
            if self.gen%20 == 0:
                print(self.gen)
                print(np.max(self.fitness))
        return self.pop

wtbounds = [-16,16]
biasbounds = [-16,16]
taubounds = [.5,10]

class MicrobialCTRNN():
    '''evolving only the base state (CTRNN parameters) with consistent HP mechanism specified in fitnessFunction '''
    def __init__(self, fitnessFunction, startpopulation, recombProb, mutatProb, generations):
        self.fitnessFunction = fitnessFunction
        self.popsize = len(startpopulation)
        self.genesize = len(startpopulation[0])
        self.CTRNNsize = int(np.sqrt(1+self.genesize)-1)
        self.cliplow = np.concatenate((np.ones(self.CTRNNsize**2)*wtbounds[0],np.ones(self.CTRNNsize)*biasbounds[0],np.ones(self.CTRNNsize)*taubounds[0]))
        self.cliphigh = np.concatenate((np.ones(self.CTRNNsize**2)*wtbounds[1],np.ones(self.CTRNNsize)*biasbounds[1],np.ones(self.CTRNNsize)*taubounds[1]))
        self.recombProb = recombProb
        self.mutatProb = mutatProb
        self.generations = generations
        self.tournaments = generations*self.popsize
        self.pop = startpopulation
        self.fitness = np.zeros(self.popsize)
        self.avgHistory = np.zeros(generations)
        self.bestHistory = np.zeros(generations)
        self.gen = 0
    def showFitness(self):
        plt.plot(self.bestHistory)
        plt.plot(self.avgHistory)
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Best and average fitness")
        plt.show()

    def fitStats(self):
        bestind = self.pop[np.argmax(self.fitness)]
        bestfit = np.max(self.fitness)
        avgfit = np.mean(self.fitness)
        self.avgHistory[self.gen]=avgfit
        self.bestHistory[self.gen]=bestfit
        return avgfit, bestfit, bestind

    def save(self,filename):
        af,bf,bi = self.fitStats()
        np.savez(filename, avghist=self.avgHistory, besthist=self.bestHistory, bestind=bi)

    def run(self):
        # Calculate all fitness once
        for i in range(self.popsize):
            self.fitness[i] = self.fitnessFunction(self.pop[i])
        # Evolutionary loop
        for g in range(self.generations):
            #print(self.pop)
            self.gen = g
            # Report statistics every generation
            self.fitStats()
            for i in range(self.popsize):
                # Step 1: Pick 2 individuals
                a = np.random.randint(0,self.popsize-1)
                b = np.random.randint(0,self.popsize-1)
                while (a==b):   # Make sure they are two different individuals
                    b = np.random.randint(0,self.popsize-1)
                # Step 2: Compare their fitness
                if (self.fitness[a] > self.fitness[b]):
                    winner = a
                    loser = b
                else:
                    winner = b
                    loser = a
                # Step 3: Transfect loser with winner --- Could be made more efficient using Numpy
                for l in range(self.genesize):
                    if (np.random.random() < self.recombProb):
                        self.pop[loser][l] = self.pop[winner][l]
                # Step 4: Mutate loser and make sure new organism stays within bounds
                self.pop[loser] += np.random.normal(0.0,self.mutatProb,size=self.genesize)
                self.pop[loser] = np.clip(self.pop[loser],self.cliplow,self.cliphigh)
                # Save fitness
                self.fitness[loser] = self.fitnessFunction(self.pop[loser])
            if self.gen%20 == 0:
                print(self.gen)
                print(np.max(self.fitness))
        return self.pop

taubbounds = [15,25]
tauwbounds = [30,50]
slidingwindowbounds = [1,1000] #in timesteps

class MicrobialHP():
    '''evolving only the HP mechanism with a fixed basal state, specified in fitnessFunction '''
    def __init__(self, fitnessFunction, startpopulation, recombProb, mutatProb, generations, differentialapp):
        self.fitnessFunction = fitnessFunction
        self.popsize = len(startpopulation)
        self.genesize = len(startpopulation[0])
        self.CTRNNsize = int((self.genesize-3)/2)
        self.cliplow = np.concatenate((np.zeros(2*self.CTRNNsize),tauwbounds[0],taubbounds[0],slidingwindowbounds[0]))
        self.cliphigh = np.concatenate((np.ones(2*self.CTRNNsize),tauwbounds[1],taubbounds[1],slidingwindowbounds[1]))
        self.recombProb = recombProb
        self.mutatProb = mutatProb
        self.generations = generations
        self.tournaments = generations*self.popsize
        self.pop = startpopulation
        self.fitness = np.zeros(self.popsize)
        self.avgHistory = np.zeros(generations)
        self.bestHistory = np.zeros(generations)
        self.bestindHistory = np.zeros((generations,self.genesize))
        self.gen = 0
    def showFitness(self):
        plt.plot(self.bestHistory)
        plt.plot(self.avgHistory)
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Best and average fitness")
        plt.show()

    def fitStats(self):
        bestind = self.pop[np.argmax(self.fitness)]
        bestfit = np.max(self.fitness)
        avgfit = np.mean(self.fitness)
        self.avgHistory[self.gen]=avgfit
        self.bestHistory[self.gen]=bestfit
        self.bestindHistory[self.gen]=bestind
        return avgfit, bestfit, bestind

    def save(self,filename):
        af,bf,bi = self.fitStats()
        np.savez(filename, avghist=self.avgHistory, besthist=self.bestHistory, bestind=bi)

    def run(self):
        # Calculate all fitness once
        for i in range(self.popsize):
            self.fitness[i] = self.fitnessFunction(self.pop[i])
        # Evolutionary loop
        for g in range(self.generations):
            #print(self.pop)
            self.gen = g
            # Report statistics every generation
            self.fitStats()
            for i in range(self.popsize):
                # Step 1: Pick 2 individuals
                a = np.random.randint(0,self.popsize-1)
                b = np.random.randint(0,self.popsize-1)
                while (a==b):   # Make sure they are two different individuals
                    b = np.random.randint(0,self.popsize-1)
                # Step 2: Compare their fitness
                if (self.fitness[a] > self.fitness[b]):
                    winner = a
                    loser = b
                else:
                    winner = b
                    loser = a
                # Step 3: Transfect loser with winner --- Could be made more efficient using Numpy
                for l in range(self.genesize):
                    if (np.random.random() < self.recombProb):
                        self.pop[loser][l] = self.pop[winner][l]
                # Step 4: Mutate loser and make sure new organism stays within bounds
                self.pop[loser] += np.random.normal(0.0,self.mutatProb,size=self.genesize)
                self.pop[loser] = np.clip(self.pop[loser],self.cliplow,self.cliphigh)
                # Save fitness
                self.fitness[loser] = self.fitnessFunction(self.pop[loser])
            if self.gen%20 == 0:
                print(self.gen)
                print(np.max(self.fitness))
        return self.pop