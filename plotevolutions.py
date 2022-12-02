import numpy as np
import matplotlib.pyplot as plt
from CTRNNclass import *
from pyloricfitness import *
import pickle

objects = []
with (open("superevol0", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
    openfile.close()

for i in range(len(objects)):
    evol = objects[i]
    print(pyloricfitness(evol['params'],debugging=True))
    plt.plot(np.arange(1,31,1),evol["best_fitness"])
    plt.plot(np.arange(1,31,1),evol["mean_fitness"])
    plt.show()
    C = CTRNN(evol['settings']['ctrnn_size'],evol['settings']['ctrnn_step_size'],400,None,evol['params'])
    C.run(0)
    C.plot()
    plt.show()

