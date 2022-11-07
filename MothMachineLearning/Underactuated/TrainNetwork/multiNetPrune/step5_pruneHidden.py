import os
import time
import pickle

import jax
import flax
import jax.numpy as jnp

import numpy as np

start_time = time.time()

'''
INPUTS
'''
modeltimestamp = '2022_11_03__03_15_19'
numParallel = 4 #you might be able to crank this up to 400 but 500 crashes after one epoch
simulationPath = 'scalarValue_1/'


dataOutput = '/home/olivia/mothPruning/mothMachineLearning_dataAndFigs/DataOutput/Experiments/pruned_bias/'
modelSubdir = os.path.join(dataOutput, modeltimestamp)

cutPercent = [0.85, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]


#np.random.seed(4206969)

act = lambda x: jnp.tanh(x) #the activation function


# EVALUATION MODE
#Initialize the weights

weightsFile = 'weights_minmax_Adam5.pkl'
weights = pickle.load(open(os.path.join(modelSubdir, weightsFile), 'rb'))

biasesFile = 'biases_minmax_Adam5.pkl'
biases = pickle.load(open(os.path.join(modelSubdir, biasesFile), 'rb'))

masksFile = 'masks_minmax_Adam5.pkl'
masks = pickle.load(open(os.path.join(modelSubdir, masksFile), 'rb'))

bmasksFile = 'bmasks_minmax_Adam5.pkl'
bmasks = pickle.load(open(os.path.join(modelSubdir, bmasksFile), 'rb'))

sparseNetsFile = 'sparseNetworks.pkl'
sparseNets = pickle.load(open(os.path.join(modelSubdir, sparseNetsFile), 'rb'))


#Load the data
X = np.load(os.path.join(simulationPath,'X.npy'))
Y = np.load(os.path.join(simulationPath, 'Y.npy'))
Xval = np.load(os.path.join(simulationPath, 'Xval.npy'))
Yval = np.load(os.path.join(simulationPath, 'Yval.npy'))
Xtest = np.load(os.path.join(simulationPath, 'Xtest.npy'))
Ytest = np.load(os.path.join(simulationPath, 'Ytest.npy'))

#Load the scalers
xMin = np.load(os.path.join(simulationPath, 'xMin.npy'))
xMax = np.load(os.path.join(simulationPath, 'xMax.npy'))
yMin = np.load(os.path.join(simulationPath, 'yMin.npy'))
yMax = np.load(os.path.join(simulationPath, 'yMax.npy'))


#Need to use the same scaler to scale the validation and test data

X = jnp.array((X-xMin)/(xMax - xMin) - 0.5)
Y = jnp.array((Y-yMin)/(yMax - yMin) - 0.5)

Xval = jnp.array((Xval-xMin)/(xMax - xMin) - 0.5)
Yval = jnp.array((Yval-yMin)/(yMax - yMin) - 0.5)

Xtest = jnp.array((Xtest-xMin)/(xMax - xMin) - 0.5)
Ytest = jnp.array((Ytest-yMin)/(yMax - yMin) - 0.5)

#for net in sparseNets:
#    print(net[0])
    
#print(len(weights[0][0]))

#weight[prune id][layer id][network id]

#print(len(weights[8]))
sweights = []
sbiases = []
smasks = []
sbmasks = []


for j in range(len(weights[0][:])):
    wlist = []
    mlist = []
    blist = []
    bmlist = []
    
    for i in range(len(sparseNets)):
        pruneID = sparseNets[i][0]
        
        w = weights[pruneID][j][i]
        m = masks[pruneID][j][i]
        b = biases[pruneID][j][i]
        bm = bmasks[pruneID][j][i]
        
        wlist.append(w)
        mlist.append(m)
        blist.append(b)
        bmlist.append(bm)
        
    W = jnp.array(wlist)
    M = jnp.array(mlist)
    B = jnp.array(blist)
    BM = jnp.array(bmlist)
    
    sweights.append(W)
    smasks.append(M)
    sbiases.append(B)
    sbmasks.append(BM)



def forward(weights, biases, masks, bmasks, inpt):
    x = jnp.einsum('ijk,lj->ilk',weights[0]*masks[0], inpt) + biases[0][:,None]*bmasks[0][:,None]
    x = act(x)
    for w,b,m,bm in zip(weights[1:-1],biases[1:-1],masks[1:-1],bmasks[1:-1]):
        x = jnp.einsum('ijk,ikl->ijl',x,w*m) + b[:,None]*bm[:,None]
        x = act(x)
    return jnp.einsum('ijk,ikl->ijl',x,weights[-1]*masks[-1]) + biases[-1][:,None]*bmasks[-1][:,None]




@jax.jit
def lossf(weights, biases, masks, bmasks, inpt, outpt):  
    xhat = forward(weights,biases,masks,bmasks,inpt)  
    loss = jnp.mean((xhat-outpt)**2)
    return loss

def lossf2(weights, biases, masks, bmasks, inpt, outpt): 
    xhat = forward(weights,biases,masks,bmasks,inpt) 
    loss = jnp.mean((xhat-outpt)**2, axis = (1,2))
    return loss



batch = 128
def epoch(optimizer, masks, bmasks):
    
    losses = []
    valLosses = []
    allNetsLosses = []
    allValLosses = []
    for i in range(len(X)//batch): #
        #print(len(X)//batch)
        #Batched training data
        x_in = X[i*batch:(i+1)*batch]
        y_in = Y[i*batch:(i+1)*batch]
        
        #Batched validation data
        xVal_in = Xval[i*batch:(i+1)*batch]
        yVal_in = Yval[i*batch:(i+1)*batch]
        
        #Calculate gradient and validation loss
        err, gr = jax.value_and_grad(lossf,argnums=(0,1))(*optimizer.target,masks,bmasks,x_in,y_in)
        valLoss = lossf(*optimizer.target,masks,bmasks,xVal_in,yVal_in)
        

        #Should this happen after the gradient update??

        #Record losses
        allNetsLoss = lossf2(*optimizer.target, masks, bmasks, x_in, y_in)
        allNetsLosses.append(allNetsLoss)
        allValLoss = lossf2(*optimizer.target, masks, bmasks, xVal_in, yVal_in)
        allValLosses.append(allValLoss)
        
        
        optimizer = optimizer.apply_gradient(gr)
        losses.append(err)
        valLosses.append(valLoss)
        print(np.array(losses).shape)
        

        #Changed to validation loss
        if ((i+1)%100) == 0:
            #print(i*batch,len(X),err)
            
            #The following is the early stopping criterion. If the loss did not improve by 1%
            #between two previous batches of 128*500 values, then terminate.
            
            #If you need higher precision maybe change 1.01 to 1.005 but that should matter much.
            
            
            #Changed to 1%
            if len(valLosses) > 1000:
                #print(np.mean(losses[-500:]), np.mean(losses[-1000:-500]), np.mean(losses[-1000:-500])/np.mean(losses[-500:]))
                if (np.mean(valLosses[-1000:-500])/np.mean(valLosses[-500:])) < 1.01:

                    break
                    
    return optimizer, np.array(losses), np.array(allNetsLosses), np.array(valLosses), np.array(allValLosses)





#The following code does the training and pruning.
#JAX (or rather FLAX) is a little weird as it stores the weights within the optimizer
#The trained weights and biases can be found in adam.target

from flax import optim
adam = optim.Adam(5E-4)
#TRAINING
adam = adam.create((sweights,sbiases))


all_errors = []
all_valErrors = []
all_weights = []
all_masks = []
all_biases = []
all_bmasks = []
seqPruneAllLosses = []
seqPruneValLosses = []


for c in cutPercent:
    
    for i in range(1):
        start = time.time()
        adam, e, allNetsLosses, vale, allValLosses  = epoch(adam, smasks, sbmasks)
        print(e.shape)
        all_errors.append(e)
        all_valErrors.append(vale)
        seqPruneAllLosses.append(allNetsLosses)
        seqPruneValLosses.append(allValLosses)
        print("EPOCH DONE", np.mean(e), c, time.time()-start)
        
    all_masks.append([np.array(x) for x in smasks])
    all_weights.append([np.array(x) for x in adam.target[0]])
    all_biases.append([np.array(x) for x in adam.target[1]])
    all_bmasks.append([np.array(x) for x in sbmasks])
    
    
    print("PRUNING")
    nmasks = []
    nbmasks = []
    count = 0
    for w,m,b,bm in zip(adam.target[0],smasks,adam.target[1],sbmasks):
        N = int(c*np.prod(w[0].shape))
        Nb = int(c*np.prod(b[0].shape))
        
        #prune only the hidden layers
        if count == 1 or 2:
            for j in range(numParallel):
                cutoff = jnp.sort(jnp.abs((w[j]*m[j]).reshape(-1)))[N]
                #cutoffb = jnp.sort(jnp.abs((b[j]*bm[j]).reshape(-1)))[Nb]
                m = m.at[j].set((jnp.abs(w[j])>cutoff)*1.0)
                #bm = bm.at[j].set((jnp.abs(b[j])>cutoff)*1.0)
        
        count += 1 
        print('Pruned to', jnp.mean(m))
        nmasks.append(m)
        #nbmasks.append(bm)
    smasks = nmasks
    #sbmasks = nbmasks
    del nmasks
    #del nbmasks
    
    
pickle.dump(all_weights, open(os.path.join(modelSubdir, 'sweights_minmax_Adam5.pkl'), 'wb'))
pickle.dump(all_biases, open(os.path.join(modelSubdir, 'sbiases_minmax_Adam5.pkl'), 'wb'))
pickle.dump(all_masks, open(os.path.join(modelSubdir,'smasks_minmax_Adam5.pkl'), 'wb'))
pickle.dump(all_bmasks, open(os.path.join(modelSubdir,'sbmasks_minmax_Adam5.pkl'), 'wb'))
pickle.dump(all_errors, open(os.path.join(modelSubdir,'serrors_minmax_Adam5.pkl'), 'wb'))
pickle.dump(all_valErrors, open(os.path.join(modelSubdir,'svalErrors_minmax_Adam5.pkl'), 'wb'))
pickle.dump(seqPruneAllLosses, open(os.path.join(modelSubdir,'slossesAllNets_Adam5.pkl'), 'wb'))
pickle.dump(seqPruneValLosses, open(os.path.join(modelSubdir,'slossesValNets_Adam5.pkl'), 'wb'))

print("--- %s seconds ---" % (time.time() - start_time))