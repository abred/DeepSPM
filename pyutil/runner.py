import calcAcc

import numpy as np
import tensorflow as tf

from collections import deque
import os
import sys
import math
import time

# Runner:
# executes training loop
# computes intermediate accuracies
# for classification
class Runner:
    def __init__(self, sess, global_step, params, numVal=1):
        self.sess = sess
        self.global_step = global_step
        self.params = params

        self.movMeanTrain = deque(maxlen=10)

        self.numVal = numVal
        self.movMeanVal = []
        for i in range(numVal):
            self.movMeanVal.append(deque(maxlen=5))
        self.movMeanVal09 = []
        for i in range(numVal):
            self.movMeanVal09.append(deque(maxlen=5))

        self.trainAccStep = 50
        self.modelSaveStep = 1000
        self.valAccStep = 250

    def init(self, out_dir, nn, trainImgCnt, valImgCnt):
        self.nn = nn
        self.out_dir = out_dir

        # init auxiliary tf stuff
        self.accuracy = tf.Variable(0., name="Accuracy")
        self.TPR = tf.Variable(0., name="TPR")
        self.TNR = tf.Variable(0., name="TNR")
        self.FPR = tf.Variable(0., name="FPR")
        self.FNR = tf.Variable(0., name="FNR")
        self.PPV = tf.Variable(0., name="PPV")
        self.NPV = tf.Variable(0., name="NPV")
        self.F1 = tf.Variable(0., name="F1")
        self.AUC = tf.Variable(0., name="AUC")
        self.sess.run([self.accuracy.initializer,
                       self.TPR.initializer,
                       self.TNR.initializer,
                       self.FPR.initializer,
                       self.FNR.initializer,
                       self.PPV.initializer,
                       self.NPV.initializer,
                       self.F1.initializer,
                       self.AUC.initializer])
        self.accSum = tf.summary.scalar("Accuracy", self.accuracy)
        self.tprSum = tf.summary.scalar("TPR", self.TPR)
        self.tnrSum = tf.summary.scalar("TNR", self.TNR)
        self.fprSum = tf.summary.scalar("FPR", self.FPR)
        self.fnrSum = tf.summary.scalar("FNR", self.FNR)
        self.npvSum = tf.summary.scalar("PPV", self.PPV)
        self.ppvSum = tf.summary.scalar("NPV", self.NPV)
        self.f1Sum = tf.summary.scalar("F1", self.F1)
        self.aucSum = tf.summary.scalar("AUC", self.AUC)
        self.summaryOpsAcc = tf.summary.merge([self.accSum,
                                               self.tprSum,
                                               self.tnrSum,
                                               self.fprSum,
                                               self.fnrSum,
                                               self.ppvSum,
                                               self.npvSum,
                                               self.f1Sum,
                                               self.aucSum])



        # different val sets
        self.writerVal = []
        for i in range(self.numVal):
            self.writerVal.append(tf.summary.FileWriter(
                os.path.join(out_dir, "val05_{}".format(i)),
                self.sess.graph))
        self.writerVal09 = []
        for i in range(self.numVal):
            self.writerVal09.append(tf.summary.FileWriter(
                os.path.join(out_dir, "val09_{}".format(i)),
                self.sess.graph))

        self.writerTrain = tf.summary.FileWriter(out_dir+'/train',
                                                 self.sess.graph)

        # number of mini batches per epoch
        self.numBatchesTrain = int(trainImgCnt / self.params['miniBatchSize'])
        self.numBatchesVal = []
        self.valImgCnt = valImgCnt
        for i in range(self.numVal):
            self.numBatchesVal.append(
                int(math.ceil(valImgCnt[i] /
                              float(self.params['miniBatchSize']))))


        # restore network weights if requested
        if self.params['classNN'] is not None:
            print("restoring class net from classNN: {}".format(
                self.params['classNN']))
            self.nn.saver.restore(
                self.sess,
                tf.train.latest_checkpoint(self.params['classNN']))
        self.saver = tf.train.Saver(max_to_keep=self.params['keepNewestModels'])
        if self.params['resume']:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(out_dir))
            print("Model restored: {}".format(tf.train.latest_checkpoint(out_dir)))

    # classification using input queues/preload
    def runQueue(self):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess,
                                                    coord=self.coord)

        try:
            step = self.sess.run(self.global_step)
            while not self.coord.should_stop():
                self.runInner(step)
                step += 1
        except tf.errors.OutOfRangeError:
            save_path = self.saver.save(self.sess,
                                        self.out_dir + "/model.ckpt",
                                        global_step=step)
            print("Model saved in file: %s" % save_path)
            print('Done training for {} epochs, {} steps.'.format(
                self.params['numTrainSteps'],
                step))
        finally:
            # When done, ask the threads to stop.
            self.coord.request_stop()

        # Wait for threads to finish.
        self.coord.join(self.threads)
        self.sess.close()

    # classification using a feed_dict
    def runLoop(self, img_train, l_train, img_val, l_val):
        self.img_train = img_train
        self.l_train = l_train
        self.img_val = img_val
        self.l_val = l_val

        for ep in range(self.params['numTrainSteps']):
            print("Train Step: {} (time: {})".format(ep, time.ctime()))
            for idx in range(self.numBatchesTrain):
                step = self.sess.run(self.global_step)
                self.runInner(step, idx=idx)
        save_path = self.saver.save(self.sess,
                                    self.out_dir + "/model.ckpt",
                                    global_step=step)
        print("Model saved in file: %s" % save_path)
        print('Done training for {} train steps, {} batches.'.format(
            self.params['numTrainSteps'],
            step))
        self.sess.close()

    # classification, inner loop, used by both variants
    def runInner(self, step, idx=None):
        # if no preload, select mini batch data
        if idx is not None:
            bidx1 = idx * self.params['miniBatchSize']
            bidx2 = (idx+1) * self.params['miniBatchSize']
            img = self.img_train[bidx1:bidx2,...]
            l = self.l_train[bidx1:bidx2,:]
            pred = self.nn.runTraining(
                inputs=(img, l))
        else:
            pred, l = self.nn.runTraining()

        # compute training accuracy for current batch?
        if step % self.trainAccStep == 1:
            acc, tpr, tnr, fpr, fnr, ppv, npv, f1, auc = \
                calcAcc.calcAccuracy(self.params, self.out_dir,
                                     pred, l,
                                     self.params['threshold'], "train")
            summary_str = self.sess.run(self.summaryOpsAcc,
                                        feed_dict={
                                            self.accuracy: acc,
                                            self.TPR: tpr,
                                            self.TNR: tnr,
                                            self.FPR: fpr,
                                            self.FNR: fnr,
                                            self.PPV: ppv,
                                            self.NPV: npv,
                                            self.F1: f1,
                                            self.AUC: auc
                                        })
            self.writerTrain.add_summary(summary_str, global_step=step)
            self.movMeanTrain.append(acc)
            print("moving mean train: {} (time: {})".format(
                sum(self.movMeanTrain)/float(len(self.movMeanTrain)),
                time.ctime()))

        # compute validation accuracy for a couple of batches?
        if step % self.valAccStep == 1 and \
           (not self.params['allForTrain'] or self.params['valInDir1'] is not None):
            for i in range(self.numVal):
                valAcc = self.computeVal(step, idx, i=i)

        # save the current weights/model?
        if step % self.modelSaveStep == 1:
            save_path = self.saver.save(
                self.sess,
                self.out_dir + "/model.ckpt",
                global_step=self.global_step)
            print("Model saved in file: {} (time: {})".format(save_path,
                                                              time.ctime()))

        step += 1
        sys.stdout.flush()

    def computeVal(self, step, idx, i=0):
        # if no preload, select mini batch data
        if idx is not None:
            idxVal = 0
            bidx1 = idxVal * self.params['miniBatchSize']
            bidx2 = (idxVal+1) * self.params['miniBatchSize']
            img = self.img_val[i][bidx1:bidx2,...]
            l = self.l_val[i][bidx1:bidx2,:]
            f = self.f_val[i][bidx1:bidx2,:]
            pred = self.nn.runPrediction(
                inputs=(img, la))
        else:
            pred, l, f = self.nn.runPrediction(i=i)

        for idxVal in range(1, self.numBatchesVal[i]):
            if idx is not None:
                bidx1 = idxVal * self.params['miniBatchSize']
                bidx2 = (idxVal+1) * self.params['miniBatchSize']
                img = self.img_val[i][bidx1:bidx2,...]
                lTmp = self.l_val[i][bidx1:bidx2,:]
                fTmp = self.f_val[i][bidx1:bidx2,:]
                predTmp = self.nn.runPrediction(
                    inputs=(img, lTmp))
            else:
                if idxVal == self.numBatchesVal[i]-1:
                    predTmp, lTmp, fTmp = self.nn.runPrediction(
                        i=i,
                        mbs=self.valImgCnt[i]%self.params['miniBatchSize'])
                else:
                    predTmp, lTmp, fTmp = self.nn.runPrediction(i=i)
            pred = np.append(pred, predTmp, axis=0)
            l = np.append(l, lTmp, axis=0)
            f = np.append(f, fTmp, axis=0)

        # fixed 0.5 threshold
        valAcc, tpr, tnr, fpr, fnr, ppv, npv, f1, auc = \
            calcAcc.calcAccuracy(self.params, self.out_dir,
                                 pred, l,
                                 threshold=0.5,
                                 prefix="val" + str(i))
        summary_str = self.sess.run(self.summaryOpsAcc,
                                    feed_dict={
                                        self.accuracy: valAcc,
                                        self.TPR: tpr,
                                        self.TNR: tnr,
                                        self.FPR: fpr,
                                        self.FNR: fnr,
                                        self.PPV: ppv,
                                        self.NPV: npv,
                                        self.F1: f1,
                                        self.AUC: auc,
                                    })
        self.writerVal[i].add_summary(summary_str, global_step=step)
        self.writerVal[i].flush()
        self.movMeanVal[i].append(valAcc)
        print("moving mean val ({}): {} (time {})".format(i,
            sum(self.movMeanVal[i])/float(len(self.movMeanVal[i])),
            time.ctime()))

        # automatically selects threshold based on roc
        valAcc, tpr, tnr, fpr, fnr, ppv, npv, f1, auc = \
            calcAcc.calcAccuracy(self.params, self.out_dir,
                                 pred, l,
                                 prefix="val" + str(i))
        # fixed 0.9 threshold
        valAcc, tpr, tnr, fpr, fnr, ppv, npv, f1, auc = \
            calcAcc.calcAccuracy(self.params, self.out_dir,
                                 pred, l,
                                 threshold=0.9,
                                 prefix="val" + str(i))
        summary_str = self.sess.run(self.summaryOpsAcc,
                                    feed_dict={
                                        self.accuracy: valAcc,
                                        self.TPR: tpr,
                                        self.TNR: tnr,
                                        self.FPR: fpr,
                                        self.FNR: fnr,
                                        self.PPV: ppv,
                                        self.NPV: npv,
                                        self.F1: f1,
                                        self.AUC: auc,
                                    })
        self.writerVal09[i].add_summary(summary_str, global_step=step)
        self.writerVal09[i].flush()
        self.movMeanVal09[i].append(valAcc)
        print("moving mean val09 ({}): {} (time {})".format(i,
            sum(self.movMeanVal09[i])/float(len(self.movMeanVal09[i])),
            time.ctime()))

        return valAcc
