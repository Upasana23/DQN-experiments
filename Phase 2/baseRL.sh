#!/bin/sh
# 
#This is a script to test job submission submit.sh
#PBS -N GenP2_TennisPong
#PBS -q copperhead
#PBS -l nodes=1:ppn=1:gpus=1:gtx1080ti,mem=15GB,walltime=150:59:59

module load tensorflow/1.10-anaconda3-cuda9.2

python /users/upattnai/Phase2/train.py

 
