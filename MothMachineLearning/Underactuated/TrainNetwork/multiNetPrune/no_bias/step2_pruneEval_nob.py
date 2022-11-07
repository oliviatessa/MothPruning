#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:06:35 2021

@author: henning
"""

import os
import time
import pickle

import jax
import flax
import jax.numpy as jnp

import numpy as np

start_time = time.time()

no_bias=True

'''
INPUTS
'''
modeltimestamp = '2022_11_01__09_29_28'
numParallel = 400 #you might be able to crank this up to 400 but 500 crashes after one epoch
simulationPath = 'scalarValue_1/'


dataOutput = '/home/olivia/mothPruning/mothMachineLearning_dataAndFigs/DataOutput/Experiments/no_bias/'
modelSubdir = os.path.join(dataOutput, modeltimestamp)

cutPercent = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90]
cutPercent = np.append(cutPercent, np.arange(0.91, 1.0, 0.01)) #the last values in cutPercent is never actually trained on


#np.random.seed(4206969)

act = lambda x: jnp.tanh(x) #the activation function


# EVALUATION MODE
#Initialize the weights

weightsFile = 'weights_minmax_Adam5.pkl'
weights = pickle.load(open(os.path.join(modelSubdir, weightsFile), 'rb'))


#biasesFile = 'biases_minmax_Adam5.pkl'
#biases = pickle.load(open(os.path.join(modelSubdir, biasesFile), 'rb'))

masksFile = 'masks_minmax_Adam5.pkl'
masks = pickle.load(open(os.path.join(modelSubdir, masksFile), 'rb'))


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



'''
This is the forward pass of numParallel many neural networks in parallel
I am using the einsums to implement batched matrix multiplication.

Apart from the the weights I am also feeding in a binary mask. All the weights
are multiplied by the mask in order to set some of the weights to 0.
'''
def forward(weights, masks, inpt): # biases,
    x = jnp.einsum('ijk,lj->ilk',weights[0]*masks[0], inpt) #+ biases[0][:,None]
    x = act(x)
    
    for w,m in zip(weights[1:-1],masks[1:-1]): #biases[1:-1], #b,
        x = jnp.einsum('ijk,ikl->ijl',x,w*m) #+ b[:,None]
        x = act(x)
    return jnp.einsum('ijk,ikl->ijl',x,weights[-1]*masks[-1]) #+ biases[-1][:,None]


'''
This is just the loss function. The @jax.jit makes everything fast.
don't forget jnp.mean in TRAINING MODE
'''
@jax.jit
def lossf(weights, masks, inpt, outpt): # biases,
    xhat = forward(weights,masks,inpt) #biases,
    loss = jnp.mean((xhat-outpt)**2)
    return loss

def lossf2(weights, masks, inpt, outpt): # biases,
    xhat = forward(weights,masks,inpt) #biases,
    loss = jnp.mean((xhat-outpt)**2, axis = (1,2))
    return loss


'''
This trains the 10 networks for one epoch. There is an early stopping criterion
that makes sure we don't run for an entire epoch if the loss doesnt improve.

The data repeats itself very quickly anyways so no need for an entire epoch.
'''
batch = 128
def epoch(optimizer, masks):
    
    losses = []
    allNetsLosses = []
    for i in range(len(X)//batch):
        print(len(X)//batch)
        x_in = X[i*batch:(i+1)*batch]
        y_in = Y[i*batch:(i+1)*batch]
        err, gr = jax.value_and_grad(lossf)(optimizer.target,masks,x_in,y_in) #* in front of optimizer #,argnums=(0,1)
        allNetsLoss = lossf2(optimizer.target, masks, x_in, y_in) #* in front of optimizer
        allNetsLosses.append(allNetsLoss)
        optimizer = optimizer.apply_gradient(gr)
        losses.append(err)
        print(np.array(losses).shape)
        

        
        if ((i+1)%100) == 0:
            #print(i*batch,len(X),err)
            '''
            The following is the early stopping criterion. If the loss did not improve by 1%
            between two previous batches of 128*500 values, then terminate.
            
            If you need higher precision maybe change 1.01 to 1.005 but that should matter much.
            '''
            if len(losses) > 1000:
                #print(np.mean(losses[-500:]), np.mean(losses[-1000:-500]), np.mean(losses[-1000:-500])/np.mean(losses[-500:]))
                if (np.mean(losses[-1000:-500])/np.mean(losses[-500:])) < 1.015:

                    break
                    
    return optimizer, np.array(losses), np.array(allNetsLosses)




'''
The following code does the training and pruning.
JAX (or rather FLAX) is a little weird as it stores the weights within the optimizer
The trained weights and biases can be found in adam.target
'''
from flax import optim
adam = optim.Adam(5E-4)

#EVALUATION
adam = adam.create((weights[0])) #,biases[0]

batch = 1024
numBatches = len(X)//batch
numVBatches = len(Xval)//batch
numTBatches = len(Xtest)//batch


all_losses = []
all_vlosses = []
all_tlosses = []

for i in range(len(cutPercent)):
    sumLoss = np.zeros(numParallel)
    sumLossVal = np.zeros(numParallel)
    sumLossTest = np.zeros(numParallel)
    
    for j in range(numBatches):
        #Batched training data
        x_in = X[i*batch:(i+1)*batch]
        y_in = Y[i*batch:(i+1)*batch]

        loss = lossf2(weights[i], masks[i], x_in, y_in) # biases[i],
        
        sumLoss += np.array(loss)
        
        print('on prune : ' + str(i) + ' and batch : ' + str(j) + ' of ' + str(numBatches))
        
    for k in range(numVBatches):
    	#Batched validation data
        xVal_in = Xval[i*batch:(i+1)*batch]
        yVal_in = Yval[i*batch:(i+1)*batch]
        
        vloss = lossf2(weights[i], masks[i], xVal_in, yVal_in) # biases[i],
        
        sumLossVal += np.array(vloss)
        
    for t in range(numTBatches):
    	#Batched test data
        xtest_in = Xtest[i*batch:(i+1)*batch]
        ytest_in = Ytest[i*batch:(i+1)*batch]
        
        tloss = lossf2(weights[i], masks[i], xtest_in, ytest_in) # biases[i],
        
        sumLossTest += np.array(tloss)
        
    finalLoss = sumLoss/numBatches
    finalvLoss = sumLossVal/numVBatches
    finaltLoss = sumLossTest/numTBatches
    
    all_losses.append(finalLoss)
    all_vlosses.append(finalvLoss)
    all_tlosses.append(finaltLoss)
    
pickle.dump(all_losses, open(os.path.join(modelSubdir, 'allPruneLosses_minmax_Adam5.pkl'), 'wb'))
pickle.dump(all_vlosses, open(os.path.join(modelSubdir, 'allvalPruneLosses_minmax_Adam5.pkl'), 'wb'))
pickle.dump(all_tlosses, open(os.path.join(modelSubdir, 'alltestPruneLosses_minmax_Adam5.pkl'), 'wb'))

print("--- %s seconds ---" % (time.time() - start_time))
