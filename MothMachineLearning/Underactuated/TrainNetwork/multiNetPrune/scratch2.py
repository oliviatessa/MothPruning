import os
import pickle

import numpy as np

modeltimestamp = '2022_11_05__01_04_10'

dataOutput = '/gscratch/dynamicsai/otthomas/MothPruning/mothMachineLearning_dataAndFigs/DataOutput/Experiments/pruned_bias/pruned_bias/'
modelSubdir = os.path.join(dataOutput, modeltimestamp)

postprunedir = os.path.join(dataOutput, modeltimestamp, 'postprune')

weightsFile = 'weights_minmax_Adam5.pkl'
weights = pickle.load(open(os.path.join(postprunedir, weightsFile), 'rb'))

biasesFile = 'biases_minmax_Adam5.pkl'
biases = pickle.load(open(os.path.join(postprunedir, biasesFile), 'rb'))

masksFile = 'masks_minmax_Adam5.pkl'
masks = pickle.load(open(os.path.join(postprunedir, masksFile), 'rb'))

bmasksFile = 'bmasks_minmax_Adam5.pkl'
bmasks = pickle.load(open(os.path.join(postprunedir, bmasksFile), 'rb'))

trainpruneLossesFile = 'allPruneLosses_minmax_Adam5.pkl'
trainpruneLosses = pickle.load(open(os.path.join(postprunedir, trainpruneLossesFile), 'rb'))

valpruneLossesFile = 'allvalPruneLosses_minmax_Adam5.pkl'
valpruneLosses = pickle.load(open(os.path.join(postprunedir, valpruneLossesFile), 'rb'))

testpruneLossesFile = 'alltestPruneLosses_minmax_Adam5.pkl'
testpruneLosses = pickle.load(open(os.path.join(postprunedir, testpruneLossesFile), 'rb'))

trainpruneLossesFile = 'allPruneLosses_minmax_Adam5.pkl'
OtrainpruneLosses = pickle.load(open(os.path.join(modelSubdir, trainpruneLossesFile), 'rb'))

valpruneLossesFile = 'allvalPruneLosses_minmax_Adam5.pkl'
OvalpruneLosses = pickle.load(open(os.path.join(modelSubdir, valpruneLossesFile), 'rb'))

testpruneLossesFile = 'alltestPruneLosses_minmax_Adam5.pkl'
OtestpruneLosses = pickle.load(open(os.path.join(modelSubdir, testpruneLossesFile), 'rb'))

print("Training loss: %s" % (trainpruneLosses[0][0]))
print("Original training loss: %s" % (OtrainpruneLosses[10][0]))

print("Validation loss: %s" % (valpruneLosses[0][0]))
print("Original validation loss: %s" % (OvalpruneLosses[10][0]))

print("Test loss: %s" % (testpruneLosses[0][0]))
print("Original test loss: %s" % (OtestpruneLosses[10][0]))
