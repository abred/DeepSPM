from multiprocessing import Process
import envServer

from distutils.dir_util import copy_tree
from random import shuffle

import sys
sys.path.append("../pyutil")
sys.path.append("..")
import signal

import parseNNArgs

import traceback
import threading
import pickle
import shutil
import glob
import os
import random
import time
import json
import math

import numpy as np
import scipy.ndimage

from dqnQNN import DQN
from replay_buffer import ReplayBuffer
from environment import Environment
import logDqn
import outDir

import tensorflow as tf
from tensorflow.python.framework import ops
# from tensorflow.python import debug as tf_debug

def printT(s):
    sys.stdout.write(s + '\n')

class dqnRunner():
    def __init__(self, sess, params, out_dir=None, agentB_sess= None):
        self.params = params
        self.sess = sess
        self.agentB_sess = agentB_sess

        self.lock = threading.Lock()
        self.modelStoreIntv = 150
        self.bufferStoreIntv = 150
        self.annealSteps = params['annealSteps']

        self.state_dim = params['pxRes']
        if self.params['verbose']:
            printT("tensorflow version: {}".format(tf.__version__))


        # create environment
        self.env = Environment(sess, params, self)
        self.numActions = self.env.numActions

        # load classifier for reward calculation
        if self.params['classNN'] is not None:
            with tf.device("/device:CPU:0"):
                self.rewardClassNet = ClassConvNetEval(self.sess, params)
                self.env.rewardClassNet = self.rewardClassNet

        # just gets or resets global_step
        self.global_step = None
        variables = tf.get_collection(
            ops.GraphKeys.GLOBAL_VARIABLES)
        for v in variables:
            if "global_step" in v.name:
                self.global_step = v
        if self.global_step is None:
            self.global_step = tf.Variable(0, name='global_step',
                                           trainable=False)
        self.resetGlStep = tf.assign(self.global_step, 0)

        # load actual dqn
        self.q = DQN(self.sess, self.params['out_dir'],
                     self.global_step, self.params, self.numActions)

        self.evalMethods= ["agent","random"]
        self.evalMethod="agent"
        self.qAgentB=None
        if (not self.params['agentB'] is None) and self.params['interEval']:
            self.qAgentB = DQN(self.agentB_sess, self.params['out_dir'],
                     self.global_step, self.params, self.numActions,agentB=True)
            self.evalMethod="agentA"
            self.evalMethods= ["agentA","random", "fixed","agentB"]
            self.sess.as_default()

        # replay buffer (size and type)
        if self.params['replaySz'] is None:
            self.replayBufferSize = 1000000
        else:
            self.replayBufferSize = self.params['replaySz']
        self.replay = ReplayBuffer(self.replayBufferSize)

        # variables for exploration decay
        self.action_step = tf.Variable(0, name='action_step',
                                       trainable=False, dtype=tf.int32)
        self.increment_ac_step_op = tf.assign(self.action_step,
                                              self.action_step+1)
        self.global_action_step = tf.Variable(0, name='global_action_step',
                                       trainable=False, dtype=tf.int32)
        self.increment_gac_step_op = tf.assign(self.global_action_step,
                                              self.global_action_step+1)
        self.episode_step = tf.Variable(0, name='episode_step',
                                        trainable=False, dtype=tf.int32)
        self.increment_ep_step_op = tf.assign(self.episode_step,
                                              self.episode_step+1)
        self.resetEpStep = tf.assign(self.episode_step, 0)
        self.resetAcStep = tf.assign(self.action_step, 0)
        self.resetGAcStep = tf.assign(self.global_action_step, 0)

        # save state
        self.saver = tf.train.Saver(max_to_keep=self.params['keepNewestModels'] )

        fn = os.path.join(self.params['out_dir'], "mainLoopTime.txt")
        self.mainLoopTimeFile = open(fn, "a")

        fn_ = os.path.join(self.params['out_dir'], "learnLoopTime.txt")
        self.learnLoopTimeFile = open(fn_, "a")


    # main function, runs the learning process
    def run(self):
        # debugging variables, for tensorboard
        if self.params['evaluation']:
            # evaluation episodes, no exploration
            eval_reward = tf.Variable(0., name="evalReward")
            eval_reward_op = tf.summary.scalar("Eval-Reward", eval_reward)
            eval_disc_reward = tf.Variable(0., name="evalDiscReward")
            eval_disc_reward_op = tf.summary.scalar("Eval-Reward_discounted",
                                                    eval_disc_reward)
            eval_stepCount = tf.Variable(0., name="evalStepCount")
            eval_stepCount_op = tf.summary.scalar("Eval-StepCount", eval_stepCount)
            eval_sum_vars = [eval_reward, eval_disc_reward, eval_stepCount]
            eval_sum_op = tf.summary.merge([eval_reward_op,
                                            eval_disc_reward_op,
                                            eval_stepCount_op])

        # (discounted) reward per episode
        episode_reward = tf.Variable(0., name="episodeReward")
        episode_reward_op = tf.summary.scalar("Reward", episode_reward)
        episode_disc_reward = tf.Variable(0., name="episodeDiscReward")
        episode_disc_reward_op = tf.summary.scalar("Reward_discounted",
                                                   episode_disc_reward)

        # average (max q)
        episode_ave_max_q = tf.Variable(0., name='epsideAvgMaxQ')
        episode_ave_max_q_op = tf.summary.scalar("Qmax_Value",
                                                 episode_ave_max_q)

        # number of steps for episode
        stepCount = tf.Variable(0., name="stepCount")
        stepCount_op = tf.summary.scalar("StepCount", stepCount)

        # number of learning iterations(total number of mini batches so far)
        global_step_op = tf.summary.scalar("GlobalStep", self.global_step)

        # current exploration epsilon
        epsilonVar = tf.Variable(0., name="epsilon")
        epsilonVar_op = tf.summary.scalar("Epsilon", epsilonVar)

        summary_vars = [episode_reward, episode_disc_reward, episode_ave_max_q,
                        stepCount, epsilonVar]
        summary_ops = tf.summary.merge([episode_reward_op,
                                        episode_disc_reward_op,
                                        episode_ave_max_q_op,
                                        stepCount_op, epsilonVar_op])
        self.writer = tf.summary.FileWriter(os.path.join(self.params['out_dir'], "train"),
                                            self.sess.graph)

        self.action_vars = []
        self.action_ops = []
        for a in range(self.numActions):
            action = tf.Variable(0., name="qval_action_" + str(a))
            action_op = tf.summary.scalar("Q-Value_Action_"+str(a), action)
            self.action_vars.append(action)
            self.action_ops.append(action_op)
        self.action_ops = tf.summary.merge(self.action_ops)

        # initialize all tensorflow variables
        # and finalize graph (cannot be modified anymore)
        self.sess.run(tf.initialize_all_variables())
        self.sess.graph.finalize()

        # for debugging, variable values before and after
        if self.params['veryveryverbose']:
            variables = tf.get_collection(
                ops.GraphKeys.GLOBAL_VARIABLES,
                scope="DQN")
            for v in variables:
                if v.name.endswith("conv1_2/weights:0"):
                    print(v.name, self.sess.run(v))

        # do we want to use pretrained weights for the dqn
        # from the classifier or a pretrained agent?
        if self.params['resume']:
            pass
        elif self.params['useClassNN']:
            print("restoring dqn net from classNN: {}".format(
                self.params['classNN']))
            if "ckpt" in self.params['classNN']:
                self.q.saver.restore(
                    self.sess,
                    self.params['classNN'])
            else:
                self.q.saver.restore(
                    self.sess,
                    tf.train.latest_checkpoint(self.params['classNN']))
        elif self.params['dqnNN'] is not None:
            print("restoring dqn net from dqnNN: {}".format(
                self.params['dqnNN']))
            if "ckpt" in self.params['dqnNN']:
                self.q.saver.restore(
                    self.sess,
                    self.params['dqnNN'])
            else:
                self.q.saver.restore(
                    self.sess,
                    tf.train.latest_checkpoint(self.params['dqnNN']))

        # main network weights are set, now run target init op
        self.sess.run(self.q.target_nn_init_op)

        if (self.params['agentB'] is not None) and self.params['interEval']:
            print("restoring agentB net from {}".format(
                self.params['agentB']))
            if "ckpt" in self.params['agentB']:
                self.qAgentB.saver.restore(
                    self.agentB_sess,
                    self.params['agentB'])
            else:
                self.qAgentB.saver.restore(
                    self.agentB_sess,
                    tf.train.latest_checkpoint(self.params['agentB']))


        # for debugging, variable values before and after
        if self.params['veryveryverbose']:
            variables = tf.get_collection(
                ops.GraphKeys.GLOBAL_VARIABLES,
                scope="DQN")
            for v in variables:
                if v.name.endswith("conv1_2/weights:0"):
                    print(v.name, self.sess.run(v))


        print("initialize classifier network")
        if self.params['classNN'] is not None:
            print("restoring reward class net from classNN: {}".format(
                self.params['classNN']))
            if "ckpt" in self.params['classNN']:
                self.rewardClassNet.saver.restore(
                    self.sess,
                    self.params['classNN'])
            else:
                self.rewardClassNet.saver.restore(
                    self.sess,
                    tf.train.latest_checkpoint(self.params['classNN']))

        # load previously trained model
        if not self.params['resume'] and self.params['loadModel']:
            if "ckpt" in self.params['loadModel']:
                self.saver.restore(
                    self.sess,
                    self.params['loadModel'])
            else:
                self.saver.restore(
                    self.sess,
                    tf.train.latest_checkpoint(self.params['loadModel']))
            printT("Model {} restored.".format(self.params['loadModel']))

        # load previously filled replay buffer
        if not self.params['resume'] and self.params['loadReplay'] is not None:

            self.replay.load(self.params['loadReplay'])

            printT("Buffer {} restored.".format(self.params['loadReplay']))

        # resume old run
        if self.params['resume']:
            self.saver.restore(sess, tf.train.latest_checkpoint(
                os.path.join(self.params['out_dir'], "models")))
            printT("Model {} restored.".format(tf.train.latest_checkpoint(
                os.path.join(self.params['out_dir'], "models"))))
           # if not self.params['interEval'] :
            self.replay.load(os.path.join(self.params['out_dir'],
                                        "replayBuffer"))
            printT("Buffer {} restored.".format(self.params['out_dir']))
        else:
            self.sess.run(self.resetGlStep)

        # start immediately for interactive test runs
        try:
            if os.environ['IS_INTERACTIVE'] == 'true' \
               and \
               not self.params['sleep']:
                self.params['startLearning'] = 1
        except KeyError:
            pass

        # exploration variables
        self.startEpsilon = self.params['epsilonStart']
        self.endEpsilon = self.params['epsilonStop']
        self.epsilon = sess.run(epsilonVar)

        # evaluation/learning/exploration
        self.evalEp = False
        self.learning = True
        self.pauseLearning = False
        self.pauseExploring = False
        self.stopLearning = False
        self.stopExploring = False

        self.qValFileExpl = open(os.path.join(self.params['out_dir'], "qValExpl.txt"), "a")
        self.qValFileEval = open(os.path.join(self.params['out_dir'], "qValEval.txt"), "a")

        self.actionLogFile = open(os.path.join(self.params['out_dir'], "actionLog.txt"), "a")
        self.episodeLogFile = open(os.path.join(self.params['out_dir'], "episodeLog.txt"), "a")
        self.episodeEvalLogFile = open(os.path.join(self.params['out_dir'], "episodeEvalLog.txt"), "a")



        # remove stop/termination file
        if os.path.exists("stop"):
            os.remove(os.path.join(params['out_dir'], "stop"))

        # reset
        if self.params['onlyLearn']:
            sess.run(self.resetEpStep)
            sess.run(self.resetAcStep)

        if self.params['onlyLearn']:
            self.learn()
            exit()

        # multi-threaded
        # learning and exploration threads act independently?
        if self.params['async']:
            t = threading.Thread(target=self.learnWrap)
            t.daemon = True
            t.start()

        if self.params['evaluation']:
            # evaluate this often
            evalEpReward = 0
            evalEpDiscReward = 0
            evalEpStepCount = 0
            evalIntv = 25
            evalCnt = 40
            evalOc = 0


        # start exploration
        self.episode = sess.run(self.episode_step)
        if self.params['verbose']:
            printT("start Episode: {}".format(self.episode))
        acs = sess.run(self.action_step)
        if self.params['verbose']:
            printT("start action step: {}".format(acs))
        self.globActStep = acs
        gacs = sess.run(self.global_action_step)
        if self.params['verbose']:
            printT("start global action step: {}".format(gacs))
        self.gac = gacs
        while self.episode<self.params['numEpisodes']:
            self.episode = sess.run(self.episode_step)
            sess.run(self.increment_ep_step_op)
            if self.params['verbose']:
                print ("STARTING NEW EPISODE:"+ str(self.episode))
            # do we want to explore/gather samples?
            while self.stopExploring:
                time.sleep(1)
            # evaluation episode (no exploration?)
            if self.params['evaluation'] and self.episode % (evalIntv+evalCnt) < evalCnt:
                self.evalEp = True
                if self.episode % (evalIntv+evalCnt) == 0:
                    if self.params['verbose']:
                        printT("Start Eval Episodes!")
                    evalOc += 1
            elif self.params['onlyLearn'] or \
               (self.params['limitExploring'] is not None \
                and self.replay.size() >= self.params['limitExploring']):
                self.pauseExploring = True
                self.evalEp = False

            else:
                self.evalEp = False

            # reset simulation/episode state
            terminal = False
            ep_reward = 0
            ep_disc_reward = 0
            ep_ave_max_q = 0
            self.inEpStep = 0


            if self.params['interEval']:
                self.evalMethod = self.evalMethods[self.episode % (len(self.evalMethods))]

            # reset environment
            # set start state and allowed actions
            nextState, allowedActions, terminal = self.env.reset(self.episode, self.evalEp, globActStep=self.globActStep)
            allowedV=self.calcAllowedActionsVector(allowedActions)

            if nextState is None:
                # unable to get state
                # restart with new episode
                continue

            lastTime=time.time()
            # step forward until terminal
            while not terminal:
                if os.path.exists(os.path.join(params['out_dir'], "stop")):
                    self.terminate()

                if self.params['async']:
                    if not t.isAlive():
                        printT("alive {}".format(t.isAlive()))
                        printT("Exception in user code:")
                        printT('-'*60)
                        traceback.print_exc(file=sys.stdout)
                        printT('-'*60)
                        sys.stdout.flush()
                        t.join(timeout=None)
                        os._exit(-1)

                # state <- nextstate
                state = nextState

                # choose action
                # random or according to dqn (depending on epsilon)
                self.inEpStep += 1
                if not self.evalEp:
                    sess.run(self.increment_ac_step_op)
                    self.globActStep += 1
                sess.run(self.increment_gac_step_op)
                self.gac += 1
                epsStep=max(0,self.globActStep-(self.params['startLearning'] /4.0) )
                tmp_step = min(epsStep, self.annealSteps)
                self.epsilon = (self.startEpsilon - self.endEpsilon) * \
                               (1 - tmp_step / self.annealSteps) + \
                               self.endEpsilon

                action = self.getActionID(state, allowedV)

                if self.evalMethod=="fixed":
                    action=self.params['fixedAction']

                # We choose a random action in these cases
                rnm=np.random.rand()
                if self.params['veryveryverbose']:
                    printT("rnm:"+str(rnm)+ " self.epsilon:"+ str(self.epsilon)+" |self.params['randomEps']:"+str(self.params['randomEps'])+" e:"+str(self.episode))
                if (self.evalMethod == "random") or (not self.pauseExploring) and (not self.evalEp) and (self.episode < self.params['randomEps'] or rnm < self.epsilon):
                    if self.params['verbose']:
                        printT("randomly selecting action")
                    action = np.random.choice(allowedActions)

                    if self.params['verbose']:
                        printT("\nEpisode: {}, Step: {}, Time:{}, Next action (e-greedy {}): {}".format(
                            self.episode,
                            self.globActStep,
                            time.ctime(),
                            self.epsilon,
                            action))
                else:  # We let the DQN choose the action
                    if self.params['verbose']:
                        printT("Greedyly selecting action:")
                    if self.params['verbose']:
                        printT("\nEpisode: {}, Step: {}, Time:{}, Next action: {}".format(
                            self.episode, self.globActStep, time.ctime(), action))

                # perform selected action and
                # get new state, reward, and termination-info
                nextState, reward, terminal, terminalP, allowedActions = self.env.act(action, self.episode, self.inEpStep ,  self.globActStep, self.evalEp)
                if self.params['veryveryverbose']:
                    print('ACTIONLOG:',str(self.globActStep),str(self.episode), str(self.inEpStep), action, self.evalEp, terminal, terminalP, reward, self.epsilon, self.evalMethod)
                self.actionLogFile.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(time.time(), str(self.globActStep),str(self.episode), str(self.inEpStep),
                                                                      action, self.evalEp, terminal, terminalP, reward, self.epsilon, self.evalMethod))
                self.actionLogFile.flush()

                allowedV=self.calcAllowedActionsVector(allowedActions)


                # accumulate episode reward
                ep_disc_reward += pow(self.params['gamma'], self.inEpStep-1) * reward
                ep_reward += reward

                if (self.evalMethod == "agent") and not self.evalEp and not self.pauseExploring:
                    self.insertSamples(np.copy(state),
                                       action, reward, terminal,
                                       np.copy(nextState),
                                       np.copy(allowedV))

                # do logging inside of one episode
                # we do not want to lose any data
                if self.params['storeModel'] and \
                  ((self.globActStep+1) % self.modelStoreIntv) == 0:
                   logDqn.logModel(self)
                if self.params['storeBuffer'] and \
                  ((self.globActStep+1) % self.bufferStoreIntv) == 0:
                   logDqn.logBuffer(self)

                # if training/exploration not decoupled, do one learning step
                if not self.params['async']:
                    for i in range(8):
                        self.learn()

                sys.stdout.flush()

                cTime=time.time()
                usedTime=cTime-lastTime

                # do we want to pause exploration thread?
                # (to simulate slower stm)
                if not self.pauseExploring and \
                   not self.evalEp and \
                   self.params['sleep'] and \
                   self.params['async'] and \
                   (self.replay.size() >= self.params['startLearning']) and \
                   (self.replay.size() >= self.params['miniBatchSize']):
                    if self.params['sleepA'] is not None:
                        sleepingTime=self.params['sleepA'] - usedTime
                        if sleepingTime >0:
                            time.sleep(sleepingTime)
                    else:
                        time.sleep(60)

                cTime=time.time()
                usedTime=cTime-lastTime
                lastTime=cTime
                self.mainLoopTimeFile.write(str(cTime)+" "+str(usedTime)+ "\n")
                self.mainLoopTimeFile.flush()


                # terminate episode after x steps
                # even if no good state has been reached
                if self.inEpStep == self.params['stepsTillTerm']:
                    self.env.switchApproachArea()
                    break
            # end episode

            # otherwise store episode summaries and print log
            if self.evalEp:
                evalEpReward += ep_reward
                evalEpDiscReward += ep_disc_reward
                evalEpStepCount += self.inEpStep
                if self.episode % (evalIntv+evalCnt) == (evalCnt-1):
                    summary_str = self.sess.run(eval_sum_op, feed_dict={
                        eval_sum_vars[0]: evalEpReward/float(evalCnt),
                        eval_sum_vars[1]: evalEpDiscReward/float(evalCnt),
                        eval_sum_vars[2]: evalEpStepCount/float(evalCnt)
                    })
                    self.writer.add_summary(summary_str, evalOc-1)
                    evalEpReward = 0.0
                    evalEpDiscReward = 0.0
                    evalEpStepCount = 0.0
                if self.params['veryveryverbose']:
                    printT("step count-eval: {}".format(self.inEpStep))
                if self.params['veryverbose']:
                    printT('Time: {} | Reward: {} | Discounted Reward: {} | Eval-Episode {}'.
                        format(time.ctime(), ep_reward, ep_disc_reward, self.episode))

                self.episodeEvalLogFile.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(time.time(), self.episode,
                                                                      ep_reward, ep_disc_reward, self.inEpStep, self.epsilon))
                self.episodeEvalLogFile.flush()
            else:
                if self.params['evaluation']:
                    et = self.episode - (evalOc * evalCnt)
                else:
                    et = self.episode
                summary_str = self.sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_disc_reward,
                    summary_vars[2]: ep_ave_max_q / float(max(self.inEpStep,1)),
                    summary_vars[3]: self.inEpStep,
                    summary_vars[4]: self.epsilon
                })
                self.writer.add_summary(summary_str, et)
                self.writer.flush()
                if self.params['veryveryverbose']:
                    printT("step count: {}".format(self.inEpStep))
                if self.params['veryveryverbose']:
                    printT('Time: {} | Reward: {} | Discounted Reward: {} | Episode {} | Buffersize: {}'.
                       format(time.ctime(), ep_reward, ep_disc_reward, self.episode,
                              self.replay.size()))

                self.episodeLogFile.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(time.time(), self.episode,
                                                        ep_reward, ep_disc_reward, self.inEpStep, self.epsilon, self.evalMethod))
                self.episodeLogFile.flush()


            # log some stuff
            if self.params['storeModel'] and \
               ((self.episode+1) % self.modelStoreIntv) == 0:
                logDqn.logModel(self)
            if self.params['storeBuffer'] and \
               ((self.episode+1) % self.bufferStoreIntv) == 0:
                logDqn.logBuffer(self)
            statsIntv = 100

            sys.stdout.flush()



        # stop learning after last episode
        self.learning = False
        sys.stdout.flush()

    def terminate(self):
        printT("terminating...........")
        sys.stdout.flush()
        self.logStuff()
        sys.stdout.flush()
        printT("EXIT NOW!")
        sys.stdout.flush()
        exit(0)

    def learnWrap(self):
        try:
            self.learn()
        except:
            printT("learn wrap failed")
            printT("Exception in user code:")
            printT('-'*60)
            traceback.print_exc(file=sys.stdout)
            printT('-'*60)
            sys.stdout.flush()
            os._exit(-1)

    def learn(self):
        y_batch = np.zeros((self.params['miniBatchSize'], 1))

        tmp = np.zeros((self.params['miniBatchSize'], self.numActions))
        lastTime=time.time()
        count=0

        while self.learning:
            # Throtteling to allow the other thread a chance
            count+=1

            cTime=time.time()
            loopTime=cTime-lastTime
            lastTime=cTime
            self.learnLoopTimeFile.write(str(cTime)+" "+str(loopTime)+ "\n")
            self.learnLoopTimeFile.flush()

            if self.stopLearning:
                time.sleep(5.0)
                continue

            if   self.replay.size() < self.params['startLearning'] or \
               self.replay.size() < self.params['miniBatchSize'] or \
               self.evalEp:
                if self.params['async']:
                    time.sleep(5.0)
                    continue
                else:
                    return

            s_batch, a_batch, r_batch, t_batch, ns_batch, allowed_batch = \
                self.replay.sample_batch(self.params['miniBatchSize'])


            if self.params['doubleDQN']:
                qValsNewState = self.estimate_ddqn(ns_batch, allowed_batch, p=False, mem=tmp)
            else:
                qValsNewState = self.predict_target_nn(ns_batch)

            for i in range(self.params['miniBatchSize']):
                if t_batch[i]:
                    y_batch[i] = r_batch[i]
                else:
                    y_batch[i] = r_batch[i] + self.params['gamma'] * qValsNewState[i]


            gS, qs, delta = self.update(s_batch, a_batch, y_batch)

            if self.params['noHardResetDQN']:
                self.update_targets()
            elif (gS+1) % self.params['resetFreq'] == 0:
                self.update_targets()

            if not self.params['async']:
                return

            if self.params['onlyLearn']:
                if (gS+1) % 1000 == 0:
                    logDqn.logModel(self)


    # Returns vector of length 'self.numActions' containing
    # Zeros for allowed actions
    # '-inf' for forbidden actions
    def calcAllowedActionsVector(self, allowedActions):
        allowedV=np.zeros(shape=(self.numActions))
        allowedV[:]=float("-inf")        # init all actions as fobidden
        for i in allowedActions:
            allowedV[i]=0               # mark actions as allowed
        return allowedV


    # get action id for max q
    def getActionID(self, state, allowedActionsV):
        if self.params['interEval'] and self.evalMethod == 'agentB':
            if self.params['verbose']:
                print("PREDICTING WITH AGENTB:")
            qs = self.qAgentB.run_predict(state)
            print(qs)
        else:
            if self.params['verbose']:
                print("PREDICTING WITH AGENT:")
            qs = self.q.run_predict(state)
        if self.evalEp:
            self.qValFileEval.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(time.time(), str(self.globActStep),str(self.episode), str(self.inEpStep), qs[0], allowedActionsV))
            self.qValFileEval.flush()
        else:
            self.qValFileExpl.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(time.time(), str(self.globActStep),str(self.episode), str(self.inEpStep), qs[0], allowedActionsV))
            self.qValFileExpl.flush()

        var_dict = {}
        for a in range(self.numActions):
            var_dict[self.action_vars[a]] = qs[0][a]
        summary_str = self.sess.run(self.action_ops, feed_dict=var_dict)
        self.writer.add_summary(summary_str, self.gac)
        self.writer.flush()
        printT("Q-values:" + str(qs))
        qs = qs + allowedActionsV
        return np.argmax(qs, axis=1)[0]


    # update dqn main network
    def update(self, states, actionIDs, targets):
        step, out, delta, loss = self.q.run_train(states, actionIDs, targets)
        # network diverged?
        if np.isnan(loss):
            printT("ABORT: NaN")
            sys.stdout.flush()
            os._exit(-1)
        return step, out, delta

    # update dqn target network
    def update_targets(self):
        self.q.run_update_target_nn()

    # estimate q values using double dqn
    # get values of target network for actions where main network is max
    def estimate_ddqn(self, states, allowedActionsV, p=False, mem=None):
        qs = self.q.run_predict(states)
        if p:
            if self.params['veryveryverbose']:
                print("allowedActionsV.shape"+ str(allowedActionsV.shape))
                print("qs.shape"+ str(qs.shape))
        qs+=allowedActionsV                         # add '-inf' to the q values of forbidden actions
        if p:
            if self.params['veryveryverbose']:
                print(states)
                print(qs.shape)
                print(states.shape)
                printT("qs: {}".format(qs))
        maxA = np.argmax(qs, axis=1)

        qs = self.q.run_predict_target(states)
        mem.fill(0)
        mem[np.arange(maxA.size), maxA] = 1
        mem = mem * qs
        mem = np.sum(mem, axis=1)
        return mem

    # predict dqns
    def predict_target_nn(self, states):
        qs = self.q.run_predict_target(states)
        return np.max(qs, axis=1)

    def predict_nn(self, states):
        qs = self.q.run_predict(states)
        return np.max(qs, axis=1)

    # insert samples into replay buffer
    def insertSamples(self, stateScaled, action, reward, terminal,
                      newStateScaled, allowedActionsV):

        stateScaled.shape = (stateScaled.shape[1],
                             stateScaled.shape[2],
                             stateScaled.shape[3])
        newStateScaled.shape = (newStateScaled.shape[1],
                                newStateScaled.shape[2],
                                newStateScaled.shape[3])

        states=(stateScaled,np.rot90(stateScaled, 2),np.fliplr(stateScaled), np.flipud(stateScaled) )
        newStates=(newStateScaled,np.rot90(newStateScaled, 2),np.fliplr(newStateScaled), np.flipud(newStateScaled) )

        if(self.params['fullAugmentation']):
            self.lock.acquire()
            for i in range(4):
                for j in range(4):
                    self.replay.add(states[i], action, reward, terminal, allowedActionsV,
                        newStates[j])
            self.lock.release()
        else:
            self.lock.acquire()
            self.replay.add(stateScaled, action, reward, terminal, allowedActionsV,
                            newStateScaled)
            self.replay.add(
                np.ascontiguousarray(np.rot90(stateScaled, 2)),
                action, reward, terminal, allowedActionsV,
                np.ascontiguousarray(np.rot90(newStateScaled, 2)))
            self.replay.add(
                np.ascontiguousarray(np.fliplr(stateScaled)),
                action, reward, terminal, allowedActionsV,
                np.ascontiguousarray(np.fliplr(newStateScaled)))
            self.replay.add(
                np.ascontiguousarray(np.flipud(stateScaled)),
                action, reward, terminal, allowedActionsV,
                np.ascontiguousarray(np.flipud(newStateScaled)))
            self.lock.release()

        # if we want to stop if buffer is full
        # or limit exploration
        if self.pauseExploring == False and \
           self.replay.size() == self.replayBufferSize:
            if self.params['termAtFull']:
                printT("Buffer FULL!")
                self.logStuff()
                self.pauseExploring = True
                # exit()
        elif self.pauseExploring == False and \
             self.params['limitExploring'] is not None and \
             self.replay.size() >= self.params['limitExploring']:
            if self.params['termAtFull']:
                printT("Buffer FULL!")
                self.logStuff()
                self.pauseExploring = True

    def logStuff(self):
        logDqn.logModel(self)
        logDqn.logBuffer(self)


if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)

    # load parameters from command line and config file
    params = parseNNArgs.parseArgs()
    if params['onlyLearn'] and \
       not params['loadReplay'] and \
       not params['loadModel']:
        print("invalid parameters! onlyLearn only avaiable in combination with loadReplay and loadModel")
        exit(-232)
    params['type'] = "agent"

    # resuming previous run?
    if params['resume']:
        out_dir = os.getcwd()
        print("resuming... {}".format(out_dir))
        newRun = False
    else:
        out_dir = outDir.setOutDir(params)
        # copy all scripts to out_dir (for potential later reuse)
        copy_tree(os.getcwd(), out_dir)
        os.makedirs(os.path.join(out_dir, "models"))
        os.makedirs(os.path.join(out_dir, "imgs"))
        os.makedirs(os.path.join(out_dir, "imgsCollect"))
        print("new start... {}".format(out_dir))
        config = json.dumps(params)
        with open(os.path.join(out_dir, "config"), 'w') as f:
            f.write(config)
        newRun = True

    params['out_dir'] = out_dir

    print("Results/Summaries/Logs will be written to: {}\n".format(out_dir))

    #pipe log to file if not in interactive mode
    interactive=False
    try:
        if os.environ['IS_INTERACTIVE'] == 'true':
            interactive=True
    except KeyError:
        pass

    if not interactive:
        print("LogFile="+ os.path.join(out_dir, "log"))
        sys.stdout.flush()
        logFile = open(os.path.join(out_dir, "log"), 'a')
        sys.stdout = sys.stderr = logFile

    if params['startServer']:
        p = Process(target=envServer.main, args=(params,))
        p.start()
        time.sleep(15)


    # add paths to load classifier later on (reward calculation)
    if params['classNN']:
        if "ckpt" not in params['classNN']:
            sys.path.insert(1, params['classNN'])
        else:
            sys.path.insert(1, os.path.dirname(params['classNN']))
        try:
            from classifierEval import ClassConvNetEval
        except:
            print("Failed to import form 'classifierEval.'")
            print("Maybe the path to your classifier net is specified wrong?")
            print(str(os.path.dirname(params['classNN'])))
            exit(-1)

    # start tensorflow session and start learning
    if params['noGPU']:
        tfconfig = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
    else:
        tfconfig = None

    if params['agentB'] is not None:
        agentB_sess_ = tf.Session()
    else:
        agentB_sess_= None

    with tf.Session(config=tfconfig) as sess:
            rl = dqnRunner(sess, params, out_dir=out_dir, agentB_sess = agentB_sess_)
            rl.run()
