import os
import numpy as np
import pandas as pd
import pickle
import glob
import sqlite3
import sys

from sklearn.model_selection import train_test_split

'''
INPUTS
'''
filename = 'mothData_s_0_9.db' #simlation you want to use
scalarValue = 'scalarValue_0_9'



simulationDir = '/home/olivia/mothPruning/MothSimulations/'
dataDir = '/home/olivia/mothPruning/MothMachineLearning/Underactuated/TrainNetwork/multiNetPrune/'

dataSubdir = os.path.join(dataDir, scalarValue)
if not os.path.exists(dataSubdir):
    os.mkdir(dataSubdir)

#Load in moth simulation
con1 = sqlite3.connect(os.path.join(simulationDir, filename))
cursorObj = con1.cursor()
res = cursorObj.execute("SELECT name FROM sqlite_master WHERE type='table';")
tableNames = [name[0] for name in res]
con1.close()

con1 = sqlite3.connect(os.path.join(simulationDir, filename))
trainDF = pd.read_sql_query("SELECT * FROM train", con1)
testDF = pd.read_sql_query("SELECT * FROM test", con1)
con1.close()


#Check if there are any repeats in the dataset 
if np.sum(trainDF.iloc[:, [16,17,18]].duplicated()) == 0:
	print('There are no repeats.')
	#0 means no repeats and we can proceed with preprocessing
	# rename columns to be consistent with other code
	trainDF.rename(columns={"x0" : "x_0", "y0" : "y_0", "phi0" : "phi_0", "theta0" : "theta_0", 
                        "x_f" : "x_99", "y_f" : "y_99", "phi_f" : "phi_99", "theta_f" : "theta_99", 
                        "xd_0" : "x_dot_0", "yd_0" : "y_dot_0", "phid_0" : "phi_dot_0", "thetad_0": "theta_dot_0", 
                        "xd_f" : "x_dot_99", "yd_f": "y_dot_99", "phid_f": "phi_dot_99", "thetad_f": "theta_dot_99", 
                        "tau0" : "tau"}, inplace=True)

	# rename columns to be consistent with other code
	testDF.rename(columns={"x0" : "x_0", "y0" : "y_0", "phi0" : "phi_0", "theta0" : "theta_0", 
                        "x_f" : "x_99", "y_f" : "y_99", "phi_f" : "phi_99", "theta_f" : "theta_99", 
                        "xd_0" : "x_dot_0", "yd_0" : "y_dot_0", "phid_0" : "phi_dot_0", "thetad_0": "theta_dot_0", 
                        "xd_f" : "x_dot_99", "yd_f": "y_dot_99", "phid_f": "phi_dot_99", "thetad_f": "theta_dot_99", 
                        "tau0" : "tau"}, inplace=True)
        
        #Convert forces to x- and y- components      
	trainDF["Fx"] = trainDF.F * np.cos(trainDF.alpha)
	trainDF["Fy"] = trainDF.F * np.sin(trainDF.alpha)

	testDF["Fx"] = testDF.F * np.cos(testDF.alpha)
	testDF["Fy"] = testDF.F * np.sin(testDF.alpha)

	# make dataset
	X = trainDF.loc[:, [ "phi_0", "theta_0", 
                    "x_99", "y_99", "phi_99", "theta_99", 
                   "x_dot_0", "y_dot_0", "phi_dot_0", "theta_dot_0"]]

	Y = trainDF.loc[:, ["Fx", "Fy", "tau", "x_dot_99", "y_dot_99", 
                    "phi_dot_99", "theta_dot_99"] ]

	# make test dataset
	Xtest = testDF.loc[:, [ "phi_0", "theta_0", 
                    "x_99", "y_99", "phi_99", "theta_99", 
                   "x_dot_0", "y_dot_0", "phi_dot_0", "theta_dot_0"]]

	Ytest = testDF.loc[:, ["Fx", "Fy", "tau", "x_dot_99", "y_dot_99", 
                    "phi_dot_99", "theta_dot_99"] ]
        
        #Getting train, val, and test from same simulation           
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state = 123)
	Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state = 123)
	X = Xtrain.to_numpy()
	Y = Ytrain.to_numpy()
	Xtest = Xtest.to_numpy()
	Ytest = Ytest.to_numpy()
	Xval = Xval.to_numpy()
	Yval = Yval.to_numpy()
	
	#Calculate scalers
	xMin = np.min(X,0)
	xMax = np.max(X,0)
	yMin = np.min(Y,0)
	yMax = np.max(Y,0)
	
	
	#Save everything 
	np.save(os.path.join(dataSubdir, 'X.npy'), X)
	np.save(os.path.join(dataSubdir, 'Y.npy'), Y)
	np.save(os.path.join(dataSubdir, 'Xtest.npy'), Xtest)
	np.save(os.path.join(dataSubdir, 'Ytest.npy'), Ytest)
	np.save(os.path.join(dataSubdir, 'Xval.npy'), Xval)
	np.save(os.path.join(dataSubdir, 'Yval.npy'), Yval)
	
	np.save(os.path.join(dataSubdir, 'xMin.npy'), xMin)
	np.save(os.path.join(dataSubdir, 'xMax.npy'), xMax)
	np.save(os.path.join(dataSubdir, 'yMin.npy'), yMin)
	np.save(os.path.join(dataSubdir, 'yMax.npy'), yMax)

	
else:
	print('There are repeats in the dataset.')
                      
