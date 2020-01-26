#!/bin/sh
#
#This is a script to test job submission submit.sh
#PBS -N IP_Tennis_Breakout
#PBS -q titan
#PBS -l nodes=1:ppn=1:gpus=1:gtx1080ti,mem=15GB,walltime=150:00:00

module load tensorflow/1.14-anaconda3-cuda10.0

python /users/upattnai/Phase3/train.py
