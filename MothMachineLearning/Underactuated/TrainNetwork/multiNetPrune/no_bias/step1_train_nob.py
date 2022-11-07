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

from datetime import datetime

start_time = time.time()


dataOutput = '/home/olivia/mothPruning/mothMachineLearning_dataAndFigs/DataOutput/Experiments/no_bias/'

#Name of experiment
modeltimestamp = datetime.now().strftime("%Y_%m_%d__%I_%M_%S")
modelSubdir = os.path.join(dataOutput, modeltimestamp)
if not os.path.exists(modelSubdir):
    os.mkdir(modelSubdir)

cutPercent = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90]
cutPercent = np.append(cutPercent, np.arange(0.91, 1.0, 0.01)) #the last values in cutPercent is never actually trained on
numUnits = [10, 400, 400, 400, 16, 7] #added input and output layer


#Number of networks you want to train in parallel
numParallel = 400

#np.random.seed(4206969)

act = lambda x: jnp.tanh(x) #the activation function


'''
This is just initialization of the weights. Usually weights are 2D tensors
with shape (input_dim, output_dim) but in this implementation they are 3D tensors
with a leading numParallel dimension that denotes networks trained in parallel
'''

# TRAINING MODE
#Initialize the weights

weights = []
#biases = []
masks = []
#I am initializing the weights and masks here
for i in range(len(numUnits)-1):
    weights.append(jnp.array(np.random.normal(0,1,(numParallel,numUnits[i],numUnits[i+1])))/5.)
    masks.append(jnp.ones((numParallel,numUnits[i],numUnits[i+1])))
    #biases.append(jnp.zeros((numParallel,numUnits[i+1])))



#Load the data and scalers
X = np.load('scalarValue_1/X.npy')
Y = np.load('scalarValue_1/Y.npy')
Xval = np.load('scalarValue_1/Xval.npy')
Yval = np.load('scalarValue_1/Yval.npy')

xMin = np.load('scalarValue_1/xMin.npy')
xMax = np.load('scalarValue_1/xMax.npy')
yMin = np.load('scalarValue_1/yMin.npy')
yMax = np.load('scalarValue_1/yMax.npy')



'''
The original code has a different scaler (min max)
Most Neural Network implementations assume 0 mean 1 std data

Min Max scalers are very prone to outliers and oftentimes not great.
If you want to have compatibility with previous runs, you can change the code to

X = jnp.array((X-np.min(X,0))/(np.max(X,0) - np.min(X,0)) - 0.5)
Y = jnp.array((Y-np.min(Y,0))/(np.max(Y,0) - np.min(Y,0)) - 0.5)
(never tested that)
'''

#Need to use the same scaler to scale the validation data
'''
Moved scaling to datapreprocess.py
'''
'''
xMin = np.min(X,0)
xMax = np.max(X,0)
yMin = np.min(Y,0)
yMax = np.max(Y,0)

pickle.dump(xMin, open(os.path.join(modelSubdir, 'xMin.pkl'), 'wb'))
pickle.dump(xMax, open(os.path.join(modelSubdir, 'xMax.pkl'), 'wb'))
pickle.dump(yMin, open(os.path.join(modelSubdir, 'yMin.pkl'), 'wb'))
pickle.dump(yMax, open(os.path.join(modelSubdir, 'yMax.pkl'), 'wb'))
'''

X = jnp.array((X-xMin)/(xMax - xMin) - 0.5)
Y = jnp.array((Y-yMin)/(yMax - yMin) - 0.5)

Xval = jnp.array((Xval-xMin)/(xMax - xMin) - 0.5)
Yval = jnp.array((Yval-yMin)/(yMax - yMin) - 0.5)


'''
This is the forward pass of numParallel many neural networks in parallel
I am using the einsums to implement batched matrix multiplication.

Apart from the the weights I am also feeding in a binary mask. All the weights
are multiplied by the mask in order to set some of the weights to 0.
'''
def forward(weights, masks, inpt): # biases,
    x = jnp.einsum('ijk,lj->ilk',weights[0]*masks[0], inpt) #+ biases[0][:,None]
    x = act(x)
    for w,m in zip(weights[1:-1],masks[1:-1]): #biases[1:-1],  #b,
        x = jnp.einsum('ijk,ikl->ijl',x,w*m) #+ b[:,None]
        x = act(x)
    return jnp.einsum('ijk,ikl->ijl',x,weights[-1]*masks[-1]) #+ biases[-1][:,None]


'''
This is just the loss function. The @jax.jit makes everything fast.
don't forget jnp.mean in TRAINING MODE
'''
@jax.jit
def lossf(weights, masks, inpt, outpt):  #biases,
    xhat = forward(weights,masks,inpt)  #biases,
    loss = jnp.mean((xhat-outpt)**2)
    return loss

def lossf2(weights, masks, inpt, outpt): # biases, 
    xhat = forward(weights,masks,inpt) #biases,
    loss = jnp.mean((xhat-outpt)**2, axis = (1,2))
    return loss


'''
This trains the 300 networks for one epoch. There is an early stopping criterion
that makes sure we don't run for an entire epoch if the loss doesnt improve.

The data repeats itself very quickly anyways so no need for an entire epoch.
'''
batch = 128
def epoch(optimizer, masks):
    
    losses = []
    valLosses = []
    allNetsLosses = []
    allValLosses = []
    for i in range(len(X)//batch):
        #print(len(X)//batch)
        #Batched training data
        x_in = X[i*batch:(i+1)*batch]
        y_in = Y[i*batch:(i+1)*batch]
        
        #Batched validation data
        xVal_in = Xval[i*batch:(i+1)*batch]
        yVal_in = Yval[i*batch:(i+1)*batch]
        
        #Calculate gradient and validation loss
        err, gr = jax.value_and_grad(lossf)(optimizer.target,masks,x_in,y_in) #* in front of optimizer #,argnums=(0,1)
        valLoss = lossf(optimizer.target,masks,xVal_in,yVal_in) #* in front of optimizer
        
        '''
        Should this happen after the gradient update??
        '''
        #Record losses
        allNetsLoss = lossf2(optimizer.target, masks, x_in, y_in)  #* in front of optimizer
        allNetsLosses.append(allNetsLoss)
        allValLoss = lossf2(optimizer.target, masks, xVal_in, yVal_in) #* in front of optimizer
        allValLosses.append(allValLoss)
        
        
        optimizer = optimizer.apply_gradient(gr)
        losses.append(err)
        valLosses.append(valLoss)
        print(np.array(losses).shape)
        

        #Changed to validation loss
        if ((i+1)%100) == 0:
            #print(i*batch,len(X),err)
            '''
            The following is the early stopping criterion. If the loss did not improve by 1%
            between two previous batches of 128*500 values, then terminate.
            
            If you need higher precision maybe change 1.01 to 1.005 but that should matter much.
            '''
            
            #Changed to 1%
            if len(valLosses) > 1000:
                #print(np.mean(losses[-500:]), np.mean(losses[-1000:-500]), np.mean(losses[-1000:-500])/np.mean(losses[-500:]))
                if (np.mean(valLosses[-1000:-500])/np.mean(valLosses[-500:])) < 1.01:

                    break
                    
    return optimizer, np.array(losses), np.array(allNetsLosses), np.array(valLosses), np.array(allValLosses)




'''
The following code does the training and pruning.
JAX (or rather FLAX) is a little weird as it stores the weights within the optimizer
The trained weights and biases can be found in adam.target
'''
from flax import optim
adam = optim.Adam(5E-4)
#TRAINING
adam = adam.create((weights)) #,biases


all_errors = []
all_valErrors = []
all_weights = []
all_masks = []
#all_biases = []
seqPruneAllLosses = []
seqPruneValLosses = []


for c in cutPercent:
    
    for i in range(1):
        start = time.time()
        adam, e, allNetsLosses, vale, allValLosses  = epoch(adam, masks)
        print(e.shape)
        all_errors.append(e)
        all_valErrors.append(vale)
        seqPruneAllLosses.append(allNetsLosses)
        seqPruneValLosses.append(allValLosses)
        print("EPOCH DONE", np.mean(e), c, time.time()-start)
        
    all_masks.append([np.array(x) for x in masks])
    all_weights.append([np.array(x) for x in adam.target]) #[0]
    #all_biases.append([np.array(x) for x in adam.target[1]])
    
    print("PRUNING")
    nmasks = []
    for w,m in zip(adam.target,masks): #adam.target[0]
        N = int(c*np.prod(w[0].shape))
        for j in range(numParallel):
            cutoff = jnp.sort(jnp.abs((w[j]*m[j]).reshape(-1)))[N]
            m = m.at[j].set((jnp.abs(w[j])>cutoff)*1.0)
        
        print('Pruned to', jnp.mean(m))
        nmasks.append(m)
    masks = nmasks
    del nmasks
    
    
pickle.dump(all_weights, open(os.path.join(modelSubdir, 'weights_minmax_Adam5.pkl'), 'wb'))
#pickle.dump(all_biases, open(os.path.join(modelSubdir, 'biases_minmax_Adam5.pkl'), 'wb'))
pickle.dump(all_masks, open(os.path.join(modelSubdir,'masks_minmax_Adam5.pkl'), 'wb'))
pickle.dump(all_errors, open(os.path.join(modelSubdir,'errors_minmax_Adam5.pkl'), 'wb'))
pickle.dump(all_valErrors, open(os.path.join(modelSubdir,'valErrors_minmax_Adam5.pkl'), 'wb'))
pickle.dump(seqPruneAllLosses, open(os.path.join(modelSubdir,'lossesAllNets_Adam5.pkl'), 'wb'))
pickle.dump(seqPruneValLosses, open(os.path.join(modelSubdir,'lossesValNets_Adam5.pkl'), 'wb'))

print("--- %s seconds ---" % (time.time() - start_time))
