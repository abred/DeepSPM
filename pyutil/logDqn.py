import time
import numpy as np
import scipy.ndimage
import random
import sys
import os
import glob

def printT(s):
    sys.stdout.write(s + '\n')

# log tensorflow weights to disk
def logModel(dqn):
    printT("Saving model... (Time: {})".format(time.ctime()))
    save_path = dqn.saver.save(dqn.sess,
                               os.path.join(dqn.params['out_dir'],
                                            "models",
                                            "model.ckpt"),
                               global_step=dqn.global_step)
    printT("Model saved in file: {} (Time: {})".format(save_path,
                                                       time.ctime()))

# log replay buffer to disk
def logBuffer(dqn):
    printT("Dumping buffer... (Time: {})".format(time.ctime()))
    dqn.replay.dump(os.path.join(dqn.params['out_dir'],
                                  "replayBuffer"))
    printT("Buffer dumped (Time: {})".format(time.ctime()))
