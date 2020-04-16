#!/bin/bash

#SBATCH --job-name=eval-classifier
#SBATCH -n 1
#SBATCH --time 0-00:10:00
#SBATCH --mem 30G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH -o log.%j

PYTHONPATH=pyutil: python \
	  classifierEval.py \
          `#__________________________________________________`\
          `# Path to model to be evaluated (supply on command line)`\
          --classNN ${1} \
          `#__________________________________________________`\
          `# Path to raw validation input data`\
	  --validationData "PATH TO VALIDATION/TEST DATA" \
          --normFactorVal1 1e-10 \
          --pxRes 64 \
          `#__________________________________________________`\
          `# or alternatively to precomputed binary blob`\
          `# (see pyutil/parseNNArgs.py for more details)`\
          `#--loadValBlob "PATH TO VALIDATION/TEST NUMPY BLOB DATA" `\
          `#__________________________________________________`\
          `# How many images to classifier at once (depends on your GPU)`\
	  --miniBatchSize 64 \
          `#__________________________________________________`\
          `# Threshold for classification probability to compute confusion matrix`\
	  --threshold 0.5
