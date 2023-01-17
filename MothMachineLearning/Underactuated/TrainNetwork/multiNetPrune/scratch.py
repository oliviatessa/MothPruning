import os
import pickle

import numpy as np

modeltimestamp = '2022_11_05__01_04_10'

dataOutput = '/gscratch/dynamicsai/otthomas/MothPruning/mothMachineLearning_dataAndFigs/DataOutput/Experiments/pruned_bias/pruned_bias/'
modelSubdir = os.path.join(dataOutput, modeltimestamp)

postprunedir = os.path.join(dataOutput, modeltimestamp, 'postprune')

weightsFile = 'weights_minmax_Adam5.pkl'
weights = pickle.load(open(os.path.join(modelSubdir, weightsFile), 'rb'))
weight = [weights[10][i][1] for i in range(5)]

biasesFile = 'biases_minmax_Adam5.pkl'
biases = pickle.load(open(os.path.join(modelSubdir, biasesFile), 'rb'))
bias = [biases[10][i][1] for i in range(5)]

masksFile = 'masks_minmax_Adam5.pkl'
masks = pickle.load(open(os.path.join(modelSubdir, masksFile), 'rb'))
mask = [masks[10][i][1] for i in range(5)]

bmasksFile = 'bmasks_minmax_Adam5.pkl'
bmasks = pickle.load(open(os.path.join(modelSubdir, bmasksFile), 'rb'))
bmask = [bmasks[10][i][1] for i in range(5)]


m = [np.append(mask[i], np.array(bmask[i]).reshape([1, len(bmask[i])]), axis=0) for i in range(5)]



FOMList = [[0,0],[0,0],[0,0],[0,0],[0,0]] #[[Num weights, num biases]] in each layer
FOM = 0
for i in range(len(m)): 
        #Calculate first-order motifs

        #Count number of connections between weights
        w_connections = np.count_nonzero(m[i][0:-1])
        FOMList[i][0] = w_connections
        #Count number of connections from bias
        b_connections = np.count_nonzero(m[i][-1])
        FOMList[i][1] = b_connections

        connections = w_connections + b_connections
        FOM += connections

print(FOM)
print(FOMList)

count = 0 
#Iterate over the masking layers in each network 
for i in range(len(m)):
    #Iterate over the columns of the mask 
    for j in range(len(m[i].T)):
        column = m[i].T[j]
        #Check to see if there are any connections between this node and the nodes in the previous layer. 
        #If there are no connections, that means there are no upstream connections and this is a ghost node. 
        n = np.count_nonzero(column)
        if n == 0:
            print('Found a ghost node: %s node in layer %s.' % (j, i))
            count += 1
            #There is no input into this node 
            #so make all downstream connections 0

            #i+1 gets us to the next mask 
            #where the jth row is the ghost node 
            m[i+1][j] = m[i+1][j] * 0 

print("Removed %s ghost nodes total." % (count))
nFOMList = [[0,0],[0,0],[0,0],[0,0],[0,0]] #[[Num weights, num biases]] in each layer
nFOM = 0
for i in range(len(m)): 
        #Calculate first-order motifs

        #Count number of connections between weights
        w_connections = np.count_nonzero(m[i][0:-1])
        nFOMList[i][0] = w_connections
        #Count number of connections from bias
        b_connections = np.count_nonzero(m[i][-1])
        nFOMList[i][1] = b_connections

        connections = w_connections + b_connections
        nFOM += connections

print(nFOM)
print(nFOMList)

mask = [m[i][:-1,:] for i in range(5)]
masks = [m.reshape(1,len(m), len(m.T)) for m in mask]

bmask = [m[i][-1,:] for i in range(5)]
bmasks = [bm.reshape(1,len(bm)) for bm in bmask]

weight = [weight[i] for i in range(5)]
weights = [w.reshape(1,len(w), len(w.T)) for w in weight]

bias = [bias[i] for i in range(5)]
biases = [b.reshape(1,len(b)) for b in bias]

masks = [masks]
bmasks = [bmasks]
weights = [weights]
biases = [biases]

pickle.dump(weights, open(os.path.join(postprunedir, 'weights_minmax_Adam5.pkl'), 'wb'))
pickle.dump(biases, open(os.path.join(postprunedir, 'biases_minmax_Adam5.pkl'), 'wb'))
pickle.dump(masks, open(os.path.join(postprunedir,'masks_minmax_Adam5.pkl'), 'wb'))
pickle.dump(bmasks, open(os.path.join(postprunedir,'bmasks_minmax_Adam5.pkl'), 'wb'))
