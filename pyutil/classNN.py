import numpy as np
from tfLearnRocAucScore import *

import tensorflow as tf
import tensorflow.contrib.slim as slim


class ClassNN(object):
    def __init__(self, sess, glStep, params):
        self.params = params
        self.sess = sess
        self.global_step = glStep
        self.dropout = params['dropout']
        self.weightDecay = params['weight-decay']
        self.learning_rate = params['learning-rate']
        if params['lr-decay']:
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate, self.global_step,
                1000, 0.9, staircase=True)

        self.momentum = params['momentum']
        self.opti = params['optimizer']

        self.keep_prob = tf.placeholder(tf.float32, shape=(),
                                        name="keep_prob_pl")
        self.isTraining = tf.placeholder(tf.bool, shape=(),
                                         name="isTraining_pl")
        if params['batchnorm']:
            self.batchnorm = slim.batch_norm
        else:
            self.batchnorm = None

        self.mbs = params['miniBatchSize']
        self.summaries = []
        self.weight_summaries = []
        self.combined_summary1 = tf.Summary()
        self.combined_summary2 = tf.Summary()
        self.combined_summary3 = tf.Summary()

        self.mbs_pl = sess.graph.get_tensor_by_name(
            "preloadClassConvNet/miniBatchSize_pl:0")

    def setSess(self, sess, out_dir):
        self.sess = sess
        self.writer = tf.summary.FileWriter(out_dir, sess.graph)

    # define loss function (cross entropy loss with optional weight decay)
    def defineLoss(self, labels):
        with tf.variable_scope('loss'):
            if self.params['aucLoss']:
                classLoss = roc_auc_score(self.params, self.nnTrain, labels)
                if self.params['veryveryverbose']:
                    classLoss = tf.Print(classLoss, [classLoss], "aucLoss ",
                                         first_n=10)
            elif self.params['penalizeFP']:
                # punish false positives?
                out = tf.sigmoid(self.nnTrain)
                # out = tf.Print(out, [out], "out ",
                #                first_n=15, summarize=1000)
                outThresh = tf.where(
                    out < tf.constant(self.params['threshold'],
                                      shape=[self.mbs, 1]),
                    tf.constant(-1.0, shape=[self.mbs, 1]),
                    tf.constant(1.0, shape=[self.mbs, 1]))
                # outThresh = tf.Print(
                #     outThresh, [outThresh], "out thresholded ",
                #     first_n=15, summarize=1000)
                labelsThresh = tf.where(
                    labels <= tf.constant(0.5,
                                          shape=[self.mbs, 1]),
                    tf.constant(-2.0, shape=[self.mbs, 1]),
                    tf.constant(1.0, shape=[self.mbs, 1]))
                # labelsThresh = tf.Print(
                #     labelsThresh, [labelsThresh], "labels thresholded ",
                #     first_n=15, summarize=1000)
                outTimesLabel = tf.multiply(outThresh, labelsThresh)
                # outTimesLabel = tf.Print(
                #     outTimesLabel, [outTimesLabel], "out * label ",
                #     first_n=15, summarize=1000)
                sampleWeights = tf.where(
                    outTimesLabel <= tf.constant(-1.5, shape=[self.mbs, 1]),
                    tf.constant(10.0, shape=[self.mbs, 1]),
                    tf.constant(1.0, shape=[self.mbs, 1]))
                if self.params['veryveryverbose']:
                    sampleWeights = tf.Print(
                        sampleWeights, [sampleWeights],
                        "sampleWeights",
                        first_n=15, summarize=1000)
                # with tf.name_scope(''):
                #     self.summaries += [
                #         tf.summary.scalar('loss_fp', outTimesLabelThreshSum)]
                classLoss = slim.losses.sigmoid_cross_entropy(
                    self.nnTrain, labels, weights=sampleWeights)
            elif self.params['relWeightPosSamples'] is not None:
                # reweight pos vs neg samples?
                sampleWeights = tf.where(
                    labels <= tf.constant(0.5,
                                          shape=[self.mbs, 1]),
                    tf.constant(1.0, shape=[self.mbs, 1]),
                    tf.constant(self.params['relWeightPosSamples'],
                                shape=[self.mbs, 1]))
                if self.params['veryveryverbose']:
                    sampleWeights = tf.Print(
                        sampleWeights, [sampleWeights],
                        "sampleWeights",
                        first_n=15, summarize=1000)
                # with tf.name_scope(''):
                #     self.summaries += [
                #         tf.summary.scalar('loss_fp', outTimesLabelThreshSum)]
                classLoss = slim.losses.sigmoid_cross_entropy(
                    self.nnTrain, labels, weights=sampleWeights)
            else:
                classLoss = slim.losses.sigmoid_cross_entropy(
                    self.nnTrain, labels)

            if self.params['veryveryverbose']:
                classLoss = tf.Print(classLoss, [classLoss], "classLoss ",
                                     first_n=10)
            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('loss_class', classLoss)]

            if self.weightDecay:
                regLoss = tf.add_n(slim.losses.get_regularization_losses())
                if self.params['veryveryverbose']:
                    regLoss = tf.Print(regLoss, [regLoss], "regLoss ", first_n=10)
                    with tf.name_scope(''):
                        self.summaries += [
                            tf.summary.scalar('loss_reg_only', regLoss)]

            if self.weightDecay:
                self.loss_op = classLoss + regLoss
            else:
                self.loss_op = classLoss

            with tf.name_scope(''):
                self.summaries += [
                    tf.summary.scalar('loss_total', self.loss_op)]

    # choose optimizer and create training op
    def defineTraining(self):
        with tf.variable_scope('train'):
            if self.opti == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                       self.momentum)
            elif self.opti == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.opti == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(
                    self.learning_rate)

            self.train_op = optimizer.minimize(self.loss_op,
                                               global_step=self.global_step)

    # run training step
    def runTraining(self, pSum=100, pLoss=50, inputs=None):
        step = self.sess.run(self.global_step)
        # if no input -> preloaded data is used
        # summaries are only computed every x step
        if inputs is None:
            if step % pSum == 0:
                _, pred, l, loss, summaries, w_summaries = \
                    self.sess.run([self.train_op,
                                   self.nnTrain,
                                   self.lTrain,
                                   self.loss_op,
                                   self.summary_op,
                                   self.weight_summary_op],
                                  feed_dict={self.keep_prob: self.dropout,
                                             self.isTraining: True})
                self.writer.add_summary(summaries, global_step=step)
                self.writer.add_summary(w_summaries, global_step=step)
                self.writer.flush()
            else:
                _, pred, l, loss = self.sess.run(
                    [self.train_op,
                     self.nnTrain,
                     self.lTrain,
                     self.loss_op],
                    feed_dict={self.keep_prob: self.dropout,
                               self.isTraining: True})
        else:
            feed_dict={
                self.images: inputs[0],
                self.labels: inputs[1],
                self.keep_prob: self.dropout,
                self.isTraining: True
            }
            if step % pSum == 0:
                _, pred, l, loss, summaries = \
                    self.sess.run([self.train_op,
                                   self.nnTrain,
                                   self.lTrain,
                                   self.loss_op,
                                   self.summary_op],
                                  feed_dict=feed_dict)
                self.writer.add_summary(summaries, global_step=step)
                self.writer.flush()
            else:
                _, pred, l, loss = self.sess.run(
                    [self.train_op,
                     self.nnTrain,
                     self.lTrain,
                     self.loss_op],
                    feed_dict=feed_dict)
        if step % pLoss == 0:
            print("Step: {}, loss: {}".format(step, loss))
        return pred, l

    # predict classification for validation images
    def runPrediction(self, pSum=150, inputs=None, i=0, mbs=None):
        step = self.sess.run(self.global_step)
        if mbs is None:
            mbs = self.params['miniBatchSize']
        if inputs is None:
            if step % pSum == 1:
                pred, l, f, sumPred = self.sess.run([self.nnVal[i],
                                                     self.lVal[i],
                                                     self.fVal[i],
                                                     self.valPredHistoSum[i]],
                                                    feed_dict={
                                            self.keep_prob: 1.0,
                                            self.isTraining: False,
                                            self.mbs_pl: mbs})
                self.writer.add_summary(sumPred, global_step=step)
                return pred, l, f
            else:
                return self.sess.run([self.nnVal[i],
                                      self.lVal[i],
                                      self.fVal[i]],
                                     feed_dict={self.keep_prob: 1.0,
                                                self.isTraining: False,
                                                self.mbs_pl: mbs})
        else:
            feed_dict={
                self.images: inputs[0],
                self.labels: inputs[1],
                self.keep_prob: 1.0,
                self.isTraining: False
            }
            return self.sess.run([self.nnVal[i]],
                                 feed_dict=feed_dict)

    # predict classification for train images
    def runPredictionTrain(self, inputs=None):
        if inputs is None:
            return self.sess.run([self.nnTrain,
                                  self.lTrain],
                                 feed_dict={self.keep_prob: 1.0,
                                            self.isTraining: False})
        else:
            feed_dict={
                self.images: inputs[0],
                self.labels: inputs[1],
                self.keep_prob: 1.0,
                self.isTraining: False
            }
            return self.sess.run([self.nnTrain],
                                 feed_dict=feed_dict)
