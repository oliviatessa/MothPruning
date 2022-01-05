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

dataOutput = '/home/olivia/mothPruning/mothMachineLearning_dataAndFigs/DataOutput/Experiments/'
modeltimestamp = '2022_01_04__04_18_29'

modelSubdir = os.path.join(dataOutput, modeltimestamp)

cutPercent = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90]
cutPercent = np.append(cutPercent, np.arange(0.91, 1.0, 0.01)) #the last values in cutPercent is never actually trained on
numUnits = [10, 800, 800, 800, 32, 7] #added input and output layer
numParallel = 10 #you might be able to crank this up to 400 but 500 crashes after one epoch

#np.random.seed(4206969)

act = lambda x: jnp.tanh(x) #the activation function


# EVALUATION MODE
#Initialize the weights

weightsFile = 'weights_minmax_Adam5_sample.pkl'
weights = pickle.load(open(os.path.join(modelSubdir, weightsFile), 'rb'))

biasesFile = 'biases_minmax_Adam5_sample.pkl'
biases = pickle.load(open(os.path.join(modelSubdir, biasesFile), 'rb'))

masksFile = 'masks_minmax_Adam5_sample.pkl'
masks = pickle.load(open(os.path.join(modelSubdir, masksFile), 'rb'))


#Load the data
X = np.load('X.npy')
Y = np.load('Y.npy')


X = jnp.array((X-np.min(X,0))/(np.max(X,0) - np.min(X,0)) - 0.5)
Y = jnp.array((Y-np.min(Y,0))/(np.max(Y,0) - np.min(Y,0)) - 0.5)


'''
This is the forward pass of numParallel many neural networks in parallel
I am using the einsums to implement batched matrix multiplication.

Apart from the the weights I am also feeding in a binary mask. All the weights
are multiplied by the mask in order to set some of the weights to 0.
'''
def forward(weights, biases, masks, inpt):
    x = jnp.einsum('ijk,lj->ilk',weights[0]*masks[0], inpt) + biases[0][:,None]
    x = act(x)
    for w,b,m in zip(weights[1:-1],biases[1:-1],masks[1:-1]):
        x = jnp.einsum('ijk,ikl->ijl',x,w*m) + b[:,None]
        x = act(x)
    return jnp.einsum('ijk,ikl->ijl',x,weights[-1]*masks[-1]) + biases[-1][:,None]


'''
This is just the loss function. The @jax.jit makes everything fast.
don't forget jnp.mean in TRAINING MODE
'''
@jax.jit
def lossf(weights, biases, masks, inpt, outpt):
    xhat = forward(weights,biases,masks,inpt)
    loss = jnp.mean((xhat-outpt)**2)
    return loss

def lossf2(weights, biases, masks, inpt, outpt):
    xhat = forward(weights,biases,masks,inpt)
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
        err, gr = jax.value_and_grad(lossf,argnums=(0,1))(*optimizer.target,masks,x_in,y_in)
        allNetsLoss = lossf2(*optimizer.target, masks, x_in, y_in)
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
                if (np.mean(losses[-1000:-500])/np.mean(losses[-500:])) < 1.01:

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
adam = adam.create((weights[0],biases[0]))

batch = 1024
numBatches = len(X)//batch
all_losses = []
for i in range(len(cutPercent)):
    sumLoss = np.zeros(numParallel)
    for j in range(numBatches):
        x_in = X[j*batch:(j+1)*batch]
        y_in = Y[j*batch:(j+1)*batch]
        loss = lossf2(weights[i], biases[i], masks[i], x_in, y_in)
        sumLoss += np.array(loss)
        print('on prune : ' + str(i) + ' and batch : ' + str(j) + ' of ' + str(numBatches))
    finalLoss = sumLoss/numBatches
    all_losses.append(finalLoss)
    
pickle.dump(all_losses, open(os.path.join(modelSubdir, 'allPruneLosses_Adam5_sample.pkl'), 'wb'))

print("--- %s seconds ---" % (time.time() - start_time))
