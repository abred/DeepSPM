import numpy as np
import os
import shutil
import sys
import time
from tensorflow.python.framework import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import defineNN

# the main DQN class
# responsible for setting up the neural network
# and executing the learning steps
class DQN:
    # number of avaiable actions
    numActions = 0
    # grey images
    col_channels = 1

    def __init__(self, sess, out_dir, glStep, params, numActions, agentB=False):
        if agentB:
            sess.as_default()
        self.sess = sess
        self.params = params
        self.summaries = []
        self.weight_summaries = []
        if params['batchnorm']:
            self.batchnorm = slim.batch_norm
        else:
            self.batchnorm = None

        self.dropout = params['dropout']
        self.weightDecay = params['weight-decay']
        self.learning_rate = params['learning-rate']
        self.momentum = params['momentum']
        self.opti = params['optimizer']
        self.tau = params['tau']
        self.state_dim = params['pxRes']

        print("dropout", self.dropout)
        self.global_step = glStep


        self.numActions = numActions

        self.scope = 'DQN'
        with tf.variable_scope(self.scope):
            self.keep_prob = tf.placeholder(tf.float32,name="keep_prob_pl")
            self.isTraining = tf.placeholder(tf.bool,name="isTraining_pl")

            # DQN Network
            self.setNN()

            self.loss_op = self.define_loss()

            if not agentB:
                self.train_op = self.defineTraining()

        # some additional tensorflow summaries (for tensorboard, debugging)
        _VARSTORE_KEY = ("__variable_store",)
        varstore = ops.get_collection(_VARSTORE_KEY)[0]
        variables = tf.get_collection(
            ops.GraphKeys.GLOBAL_VARIABLES,
            scope=self.scope)

        for v in variables:
            if params['veryveryverbose']:
                print(v.name)
            if v.name.endswith("weights:0") or \
               v.name.endswith("biases:0"):
                s = []
                var = v
                mean = tf.reduce_mean(var)
                s.append(tf.summary.scalar(v.name+'mean', mean))
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                s.append(tf.summary.scalar(v.name+'stddev', stddev))
                s.append(tf.summary.scalar(v.name+'max', tf.reduce_max(var)))
                s.append(tf.summary.scalar(v.name+'min', tf.reduce_min(var)))
                s.append(tf.summary.histogram(v.name+'histogram', var))

                self.weight_summaries += s

        self.summary_op = tf.summary.merge(self.summaries)
        self.weight_summary_op = tf.summary.merge(self.weight_summaries)
        self.writer = tf.summary.FileWriter(out_dir, sess.graph)

        # necessary if we want to initialize the weights
        # with pretrained values from the ranking network
        if params['useClassNN'] or params['dqnNN'] is not None:
            if params['useClassNN']:
                newScope = "classConvNet"
            else:
                newScope = "DQN"
            vd = {}
            for v in variables:
                if params['veryveryverbose']:
                    print(v.name)
                vn = v.name.split(":")[0]
                if (vn.endswith("weights") or \
                    vn.endswith("biases") or \
                    "BatchNorm" in vn) and \
                    not "train" in vn and \
                    not "out" in vn:
                    vn = vn.replace(self.scope, newScope)
                    if not "target" in vn:
                        if params['veryveryverbose']:
                            print(vn)
                        vd[vn] = v
            self.saver = tf.train.Saver(vd, max_to_keep=200)
        else:
            self.saver = tf.train.Saver(max_to_keep=200)

    # setup and define main and target networks
    # otherwise: use images as input
    def setNN(self):
        prevTrainVarCount = len(tf.trainable_variables())
        prevTotalVarCount = len(tf.global_variables())

        # main network
        self.input_pl = tf.placeholder(
            tf.float32,
            shape=[None,
                   self.state_dim,
                   self.state_dim,
                   self.col_channels],
            name='input')
        self.nn = defineNN.defineNN(self, self.input_pl,
                                    numOut=self.numActions,
                                    isDQN=True)
        self.nn_train_params = tf.trainable_variables()[prevTrainVarCount:]
        self.nn_total_params = tf.global_variables()[prevTotalVarCount:]

        # Target Network
        with tf.variable_scope('target'):
            prevTrainVarCount = len(tf.trainable_variables())
            prevTotalVarCount = len(tf.global_variables())

            self.target_input_pl = tf.placeholder(
                tf.float32,
                shape=[None,
                       self.state_dim,
                       self.state_dim,
                       self.col_channels],
                name='input')
            self.target_nn = defineNN.defineNN(self,
                                               self.target_input_pl,
                                               numOut=self.numActions,
                                               isTargetNN=True,
                                               isDQN=True)
            self.target_nn_train_params = \
                tf.trainable_variables()[prevTrainVarCount:]
            self.target_nn_total_params = \
                tf.global_variables()[prevTotalVarCount:]

            self.target_nn_init_op = self.define_init_target_nn_op()
            self.target_nn_update_op = self.define_update_target_nn_op()


    # define tensorflow op to update the target network
    # hard copy every X steps or rolling update
    def define_update_target_nn_op(self):
        with tf.variable_scope('update'):
            if self.params['noHardResetDQN']:
                tau = tf.constant(self.tau, name='tau')
                invtau = tf.constant(1.0-self.tau, name='invtau')
                return \
                    [self.target_nn_total_params[i].assign(
                        tf.multiply(self.nn_total_params[i], tau) +
                        tf.multiply(self.target_nn_total_params[i], invtau))
                     for i in range(len(self.target_nn_total_params))]
            else:
                return \
                    [self.target_nn_total_params[i].assign(self.nn_total_params[i])
                     for i in range(len(self.target_nn_total_params))]

    # define tensorflow op to initialize the target network
    # (with the values of the main network)
    def define_init_target_nn_op(self):
        with tf.variable_scope('update'):
            return \
                [self.target_nn_total_params[i].assign(self.nn_total_params[i])
                 for i in range(len(self.target_nn_total_params))]

    # define the MSE loss function
    def define_loss(self):
        with tf.variable_scope('loss2'):
            self.td_targets_pl = tf.placeholder(tf.float32, [None, 1],
                                                name='tdTargets')
            self.action_ids_pl = tf.placeholder(tf.int32, [None, 1],
                                                name='actionIDs')

            index_mask = tf.reshape(tf.one_hot(self.action_ids_pl,
                                               self.numActions),
                                    [-1, self.numActions])
            qs = tf.reduce_sum(self.nn * index_mask,
                                    axis=1, keep_dims=True)
            targets = self.td_targets_pl

            self.delta = targets - qs

            if self.params['huberLoss']:
                d = 1.0
                lossL2 = tf.where(tf.abs(self.delta) < d,
                                  0.5 * tf.square(self.delta),
                                  0.5*(d*d)+d*(tf.abs(self.delta)-d),
                                  name='clipped_error')
            else: # MSE loss
                lossL2 = slim.losses.mean_squared_error(
                    targets,
                    qs)
            if self.params['veryveryverbose']:
                lossL2 = tf.Print(lossL2, [lossL2], "lossL2 ", first_n=25)

            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('mean_squared_diff_loss',
                                      lossL2)]
            if self.weightDecay != 0.0:
                lossReg = tf.add_n(slim.losses.get_regularization_losses(
                    self.scope))
                if self.params['veryveryverbose']:
                    lossReg = tf.Print(lossReg, [lossReg], "regLoss ", first_n=25)
                with tf.name_scope(''):
                    self.summaries += [
                        tf.summary.scalar('mean_squared_diff_loss_reg',
                                          lossReg)]
                loss = lossL2 + lossReg
            else:
                loss = lossL2

            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('mean_squared_diff_loss_with_reg',
                                      loss)]

        return loss

    # define the tensorflow training op
    # select optimizer
    def defineTraining(self):
        with tf.variable_scope('train'):
            learning_rate = self.learning_rate
            if self.params['lr-decay']:
                learning_rate = tf.train.exponential_decay(
                    self.learning_rate, self.global_step,
                    5000, 0.9, staircase=False)
            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('learning_rate',
                                      learning_rate)]
            momentum = self.momentum
            if self.params['mom-decay']:
                momentum = 1.0 - tf.train.exponential_decay(
                    1.0 - self.momentum, self.global_step,
                    50000, 0.9, staircase=False)
            if self.params['veryveryverbose']:
                learning_rate = tf.Print(learning_rate, [learning_rate],
                                         "learning_rate", first_n=15)
            if self.opti == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                       momentum)
            elif self.opti == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate)
            elif self.opti == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate)

            print(self.optimizer)
            return self.optimizer.minimize(self.loss_op,
                                           global_step=self.global_step)

    # run learning step (for main network)
    def run_train(self, inputs, actionIDs, targets):
        step = self.sess.run(self.global_step)
        wSum = 250
        lSum = 50
        if (step+1) % wSum == 0:
            out, delta, loss, _, summaries = self.sess.run(
                [self.nn,
                 self.delta,
                 self.loss_op,
                 self.train_op,
                 self.weight_summary_op],
                feed_dict={
                    self.input_pl: inputs,
                    self.action_ids_pl: actionIDs,
                    self.td_targets_pl: targets,
                    self.isTraining: True,
                    self.keep_prob: self.dropout
                })
            self.writer.add_summary(summaries, step)
            self.writer.flush()
        elif (step+1) % lSum == 0:
            out, delta, loss, _, summaries = self.sess.run(
                [self.nn,
                 self.delta,
                 self.loss_op,
                 self.train_op,
                 self.summary_op],
                feed_dict={
                    self.input_pl: inputs,
                    self.action_ids_pl: actionIDs,
                    self.td_targets_pl: targets,
                    self.isTraining: True,
                    self.keep_prob: self.dropout
                })
            self.writer.add_summary(summaries, step)
            self.writer.flush()
        else:
            out, delta, loss, _ = self.sess.run([self.nn,
                                                 self.delta,
                                                 self.loss_op,
                                                 self.train_op],
                                                feed_dict={
                self.input_pl: inputs,
                self.action_ids_pl: actionIDs,
                self.td_targets_pl: targets,
                self.isTraining: True,
                self.keep_prob: self.dropout
            })
        return step, out, delta, loss

    # run evaluation for main network
    def run_predict(self, inputs):
        return self.sess.run(self.nn, feed_dict={
            self.input_pl: inputs,
            self.isTraining: False,
            self.keep_prob: 1.0
        })

    # run evaluation for target network
    def run_predict_target(self, inputs):
        if self.params['updateTargetBNStatsWithTau']:
            isTrain = False
        else:
            isTrain = True

        return self.sess.run(self.target_nn, feed_dict={
            self.target_input_pl: inputs,
            self.isTraining: isTrain,
            self.keep_prob: 1.0
        })

    # run update op for target network
    def run_update_target_nn(self):
        self.sess.run(self.target_nn_update_op)
