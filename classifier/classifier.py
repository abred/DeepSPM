from distutils.dir_util import copy_tree

import json
import numpy as np
import os
import sys
import time

import preload
import parseNNArgs
import loadData as ld
import classNN
import runner
import outDir
import defineNN

from tensorflow.python.framework import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim


class ClassConvNet(classNN.ClassNN):
    def __init__(self, sess, out_dir, glStep,
                 trainImages,
                 trainLabels,
                 valImages,
                 valLabels,
                 valFiles,
                 params):
        super(ClassConvNet, self).__init__(sess, glStep, params)
        self.state_dim = params['pxRes']

        # create network and define loss function
        # depends on if preloaded data and queues or a feed_dict is used
        if params['preload']:
            with tf.variable_scope("classConvNet") as scope:
                self.nnTrain = defineNN.defineNN(self, trainImages)
                self.nn_params = tf.trainable_variables()
                scope.reuse_variables()
                self.nnVal = []
                for i in range(len(valImages)):
                    self.nnVal.append(defineNN.defineNN(self, valImages[i]))

            self.lTrain = trainLabels
            self.defineLoss(trainLabels)
            self.defineTraining()
            self.lVal = valLabels
            self.images = trainImages
        else:
            self.images = tf.placeholder(
                tf.float32,
                shape=(None,
                       self.state_dim,
                       self.state_dim,
                       1),
                name='input')
            self.nn = defineNN.defineNN(self, self.images)
            self.nn_params = tf.trainable_variables()
            self.nnTrain = self.nn
            self.nnVal = self.nn

            self.labels = tf.placeholder(tf.float32, [None, 1],
                                         name='labels')
            self.lTrain = self.labels
            self.defineLoss(self.labels)
            self.defineTraining()
            self.lVal = self.labels

        # some additional tensorflow summaries (for tensorboard, debugging)
        _VARSTORE_KEY = ("__variable_store",)
        varstore = ops.get_collection(_VARSTORE_KEY)[0]
        variables = tf.get_collection(
            ops.GraphKeys.GLOBAL_VARIABLES,
            scope="classConvNet")

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
                if params['veryveryverbose']:
                    print("weight sum", v.name)

        self.fVal = valFiles

        if params['veryveryverbose']:
            print(set(self.nn_params).symmetric_difference(set(tf.trainable_variables())))
        self.summary_op = tf.summary.merge(self.summaries)
        self.weight_summary_op = tf.summary.merge(self.weight_summaries)
        self.valPredHistoSum = []
        for i in range(len(self.nnVal)):
            self.valPredHistoSum.append(tf.summary.histogram('valPredHisto',
                                                              self.nnVal[i]))

        # necessary if we want to initialize the weights
        # with pretrained values
        if params['classNN'] is not None:
            self.saver = tf.train.Saver()


def loadData(params, newRun):
    # load or restore data
    print("start loading data, time: {}".format(time.ctime()))
    if params['loadBlob'] is not None:
        img_train, l_train, f_train, \
            img_val, l_val, f_val = ld.restoreData(params['loadBlob'])
    elif newRun:
        img_train, l_train, f_train, \
            img_val, l_val, f_val = ld.loadData(params['out_dir'],
                                                         params)
    else:
        img_train, l_train, f_train, \
            img_val, l_val, f_val = ld.restoreData(params['out_dir'])

    print("end loading data, time: {}".format(time.ctime()))
    print("Train images shape", img_train.shape, l_train.shape)
    print("Train images min/max", img_train.min(), img_train.max())
    print("Train images data type  ", img_train.dtype)
    for i in range(len(img_val)):
        print("Val  images shape", img_val[i].shape, l_val[i].shape)
        print("Val  images data type  ", img_val[i].dtype)

    return img_train, l_train, f_train, img_val, l_val, f_val


if __name__ == "__main__":
    # load parameters from command line and config file
    params = parseNNArgs.parseArgs()
    params['type'] = "classifier"

    # resuming previous run?
    if params['resume']:
        out_dir = os.getcwd()
        print("resuming... {}".format(out_dir))
        newRun = False
    else:
        out_dir = outDir.setOutDir(params)
        # copy all scripts to out_dir (for potential later reuse)
        copy_tree(os.getcwd(), out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print("new start... {}".format(out_dir))
        config = json.dumps(params)
        with open(os.path.join(out_dir, "config"), 'w') as f:
            f.write(config)
        newRun = True
    params['out_dir'] = out_dir

    print("Results/Summaries/Logs will be written to: {}\n".format(out_dir))

    # pipe log to file if not in interactive mode
    try:
        if os.environ['IS_INTERACTIVE'] != 'true':
            sys.stdout.flush()
            logFile = open(os.path.join(out_dir, "log"), 'a')
            sys.stdout = sys.stderr = logFile
    except KeyError:
        pass

    # start tensorflow session
    global_step = tf.Variable(1, name='global_stepClass',
                              trainable=False)
    sess = tf.Session()

    # load data
    img_train, l_train, f_train, \
        img_val, l_val, f_val = loadData(params, newRun)

    # preload data to gpu memory?
    if params['preload']:
        # create input queues etc...
        with tf.variable_scope("preloadClassConvNet") as scope:
            pl = preload.Preloader(img_train, l_train,
                                   img_val, l_val, f_val)
            images_train, labels_train, \
            images_val, labels_val, files_val = \
                pl.setupPreload(params)

        # create tensorflow network
        nn = ClassConvNet(sess, out_dir, global_step,
                          images_train, labels_train,
                          images_val, labels_val, files_val,
                          params)
        # upload data to gpu
        pl.initPreload(sess)
    else:
        # use simple feed dict instead of queues (simpler but less efficient)
        # create tensorflow network
        nn = ClassConvNet(sess, out_dir, global_step,
                          None, None, None, None, None,
                          params)

    # number of train and val images
    trainImgCnt = img_train.shape[0]
    # if params['allForTrain']:
    #     valImgCnt = 0
    # else:
    valImgCnt = []
    for i in range(len(img_val)):
        valImgCnt.append(img_val[i].shape[0])


    # this runner organizes the actual training/evaluation
    # if params['allForTrain']:
    #     op = runner.Runner(sess, global_step, params, numVal=0)
    # else:
    op = runner.Runner(sess, global_step, params, numVal=len(img_val))

    # initialize tensorflow variables (weights etc)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)

    # init runner variables (accuracies, summaries etc)
    op.init(out_dir, nn, trainImgCnt, valImgCnt)

    # set session used by tensorflow network
    nn.setSess(sess, out_dir)

    # start training
    if params['preload']:
        op.runQueue()
    else:
        op.runLoop(img_train, l_train, img_val, l_val, transf=True)
