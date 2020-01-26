#!/bin/sh
# 
#This is a script to test job submission submit.sh
#PBS -N ipRLBreakout7
#PBS -q titan
#PBS -l nodes=1:ppn=1:gpus=1:gtx1080ti,mem=15GB,walltime=165:59:59

module load tensorflow/1.7-anaconda3-cuda9

python /users/vkanchar/thesis/IterativePracticeRL_atari/train.py

 
