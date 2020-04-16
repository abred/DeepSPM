#!/bin/bash

#SBATCH --job-name=classifier
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --time 0-02:00:00
#SBATCH --mem 20G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH -o log.%j
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu2

if [ "x1" != "x${1}" ]
then
    resume=0
else
    resume=1
fi

# for all possible args see pyutil/parseNNArgs.py
# use prepareAndRun.sh to start this script (first copies all scripts to new folder and then executes them on cluster)
PYTHONPATH=pyutil: python \
	  classifier.py \
          --version 0.1 \
          `#__________________________________________________`\
          `# We need this to resume using the resume.sh script`\
          `# (set automatically, do not change this)`\
          --resume "${resume}" \
          `#__________________________________________________`\
          `# Describing the CNN architecture for the classifier`\
          --cnnSetSizes "64,128,256,512" \
          --fcSizes "1024,1024" \
          `#__________________________________________________`\
          `# Path to training data and normalization factor`\
          -i "PATH TO TRAINING IMAGES" \
          --normFactorTrain 1e-10 \
          `#__________________________________________________`\
          `# Path to validation data and normalization factor`\
          --valInDir1 "PATH TO VALIDATION/TEST DATA" \
          --normFactorVal1 1e-10 \
          `#__________________________________________________`\
          `# Don't split data provided with -i in train and validation set`\
          `# (if you provide separate validation data with valInDirX`\
          --allForTrain \
          `#__________________________________________________`\
          `# Training parameters`\
          --relWeightPosSamples 3 \
          --pxRes 64 \
          --miniBatchSize 64 \
          --dropout 0.5 \
          --learning-rate 0.001 \
          --optimizer adam \
          --threshold 0.5 \
          --weight-decay 0.00005 \
          --numTrainSteps 50000 \
          --batchnorm \
          `#__________________________________________________`\
          `# Apply augmentation`\
          --distorted \
