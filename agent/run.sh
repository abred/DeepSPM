#!/bin/bash

#SBATCH --job-name=agent
#SBATCH -n 1
#SBATCH --time 0-0:30:00
#SBATCH --mem 30G
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH -o log.%j
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu2

if [[ $# -eq 2 ]] ; then
  restoreApproachAreaMask=${2}
else
  restoreApproachAreaMask=""
fi

if [ "x1" != "x${1}" ]
then
    resume=0
else
    resume=1
fi

# for all possible args see pyutil/parseNNArgs.py
# use prepareAndRunJug.sh to start this script (first copies all scripts to new folder and then executes them)
PYTHONPATH=pyutil: python \
            `#-m cProfile -o profOut -s 'cumtime' `\
            agent.py \
            --version 0.1 \
            `#__________________________________________________`\
            `# No random episodes at the beginning.`\
            --randomEps 0 \
            `#__________________________________________________`\
            `# If the probe cannot be restored for 100 steps the episode terminates.`\
            --stepsTillTerm 100 \
            `# We start learning when the buffer contains 2000 elements.`\
            --startLearning 2000 \
            `#__________________________________________________`\
            `# Terminate after this nu,ber of episodes`\
            --numEpisodes 2000000 \
            `#__________________________________________________`\
            `# Separate threads for exploration and learning`\
            --async \
            `#__________________________________________________`\
            `# At this address we try to connect to the server`\
            --host "localhost" \
            --port 50009 \
            `#__________________________________________________`\
            `# We start a dummy server`\
            --startServer \
            `#__________________________________________________`\
            `# The dummy server will return random images from this location`\
            --dummyImageDir "PATH TO DUMMY IMAGES" \
            `#__________________________________________________`\
            `# We define how much output we want printed`\
            --verbose \
            `#__________________________________________________`\
            `# The size of the images we scan in pixels`\
            --pxRes 64 \
            `#__________________________________________________`\
            `# Training parameters`\
            --learning-rate 0.0005 \
            --momentum 0.9 \
            --optimizer adam \
            --miniBatchSize 64 \
            --dropout 0.5 \
            --mom-decay \
            --storeModel \
            `#__________________________________________________`\
            `# Reinforcement learning parameters`\
            --gamma 0.95 \
            --reward stepF \
            --rewardPos -1 \
            --rewardFinal 10 \
            --doubleDQN \
            --resetFreq 100 \
            --replaySz 15000 \
            --storeBuffer \
            `#__________________________________________________`\
            `# Describing the decay epsilon for our epsilon greedy strategy`\
            --epsilonStart 1.0 \
            --epsilonStop 0.05 \
            --annealSteps 500 \
            `#__________________________________________________`\
            `# This should point to the the classifier directory`\
            --classNN "PATH TO CLASSIFIER NETWORK" \
            `#__________________________________________________`\
            `# We use the classifier also for initialization`\
            --useClassNN \
            `#__________________________________________________`\
            `# Describing the CNN architecture for the classifier and the agent`\
            --cnnSetSizes "64,128,256,512" \
            --fcSizes "1024,1024" \
            --batchnorm \
            --cnnSetSizesDQN "64,128,256,512" \
            --fcSizesDQN "1024,1024" \
            --batchnormDQN \
            `#__________________________________________________`\
            `# The action file describes all probe shaping actions that can be chosen by the agent`\
            --actionFile "./actions.csv" \
            `#__________________________________________________`\
            `# The probability threshold for considering the probe as good`\
            --threshold 0.9 \
            `#__________________________________________________`\
            `# We need this to resume using the resume.sh script`\
            `# (set automatically, do not change this)`\
            --resume "${resume}" \
            `#__________________________________________________`\
            `# This is required to remember which part of the current approach area are still free`\
            `# in case you resume the operation after stopping`\
            --restoreApproachAreaMask "${restoreApproachAreaMask}" \
            --keepNewestModels 2 \
            `#__________________________________________________`\
            `# IMPORTANT: this determines how many consecutive images have to be classified as 'bad probe'`\
            `# before starting a conditioning episode`\
            `# Set this to 'None' for training. Set it to positive number during operation.`\
            --maxBadImgCount 10 \
