"""
adapted from:
Data structure for implementing experience replay
Author: Patrick Emami

License:
The MIT License (MIT)

Copyright (c) 2016 Patrick E.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import random
from collections import deque

import numpy as np
import pickle
import sys
import scipy.ndimage
import gc
import glob
import os
from natsort import natsorted

# stores the experience gathered by the agent
class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=None):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.oldPos = 0
        self.currPos = 0
        self.full = False
        self.buffer = []
        self.featCount = 3
        random.seed(random_seed)
        self.useSubBuffer = False

    def add(self, s, a, r, t, allowedActionV, s2):
        experience = (s, a, r, t, s2, allowedActionV)
        self.buffer.append(experience)
        self.count += 1
        self.currPos = (self.currPos + 1)

    def size(self):
        return self.count

    # uniform random sampling
    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            if self.useSubBuffer:
                batch = random.sample(self.buffer[:4000], self.count)
            else:
                batch = random.sample(self.buffer, self.count)
        else:
            if self.useSubBuffer:
                batch = random.sample(self.buffer[:4000], batch_size)
            else:
                batch = random.sample(self.buffer[-self.buffer_size:], batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.reshape(np.array([_[1] for _ in batch]), (batch_size, 1))
        r_batch = np.reshape(np.array([_[2] for _ in batch]), (batch_size, 1))
        t_batch = np.reshape(np.array([_[3] for _ in batch]), (batch_size, 1))
        s2_batch = np.array([_[4] for _ in batch])
        allowed_batch = np.array([_[5] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch, allowed_batch

    def clear(self):
        self.buffer = []
        self.full = False
        self.count = 0
        self.oldPos = 0
        self.currPos = 0

    def dump(self, fn):
        with open(fn +
                  "from" + str(self.oldPos) +
                  "to" + str(self.currPos) +
                  ".pickle", 'wb') as f:
            gc.disable()
            pickle.dump(self.buffer[self.oldPos:self.currPos], f)
            gc.enable()
        self.oldPos = self.currPos

    def load(self, fn):
        fls = natsorted(glob.glob(fn + "*.pickle"))
        for fl in fls:
            with open(fl, 'rb') as f:
                self.buffer += pickle.load(f)
            self.currPos = int(
                os.path.basename(fl).split("to")[1].split(".pickle")[0])
            self.oldPos = self.currPos
            print("loaded buffer part {} {}".format(fl, self.currPos))
        self.count = len(self.buffer)
        print("buffer size {}".format(self.count))
