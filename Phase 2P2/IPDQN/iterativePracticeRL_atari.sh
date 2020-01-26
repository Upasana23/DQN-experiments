#!/bin/sh
#
#This is a script to test job submission submit.sh
#PBS -N GenIP_Boxing_Breakout
#PBS -q copperhead
#PBS -l nodes=1:ppn=1:gpus=1:gtx1080ti,mem=15GB,walltime=120:59:59

module load tensorflow/1.10-anaconda3-cuda9.2

python /users/upattnai/Phase2P2/IPDQN/train.py
