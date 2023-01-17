#!/bin/bash

#SBATCH --job-name=test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=otthomas@uw.edu

#SBATCH --account=dynamicsai
#SBATCH --partition=gpu-a40 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8 
#SBATCH --mem=64G 
#SBATCH --gpus=1  
#SBATCH --time=00-12:00:00 

#SBATCH --chdir=/gscratch/dynamicsai/otthomas/MothPruning/MothMachineLearning/Underactuated/TrainNetwork/multiNetPrune/
#SBATCH --export=all
#SBATCH --output=/gscratch/dynamicsai/otthomas/MothPruning/mothMachineLearning_dataAndFigs/DataOutput/slurmOutputs/test.out
#SBATCH --error=/gscratch/dynamicsai/otthomas/MothPruning/mothMachineLearning_dataAndFigs/DataOutput/slurmErrors/test.err

python3 /mmfs1/gscratch/dynamicsai/otthomas/MothPruning/MothMachineLearning/Underactuated/TrainNetwork/multiNetPrune/testf.py