import numpy as np
import os
import sys
import time

import parseNNArgs
import loadData as ld
import calcAcc

import tensorflow as tf
import tensorflow.contrib.slim as slim


class ClassConvNetEval:

    def __init__(self, sess, params, crop=False):
        self.sess = sess
        self.params = params
        self.dropout = params['dropout']
        self.weightDecay = params['weight-decay']
        if params['batchnorm']:
            self.batchnorm = slim.batch_norm
        else:
            self.batchnorm = None

        self.summaries = []
        self.weight_summaries = []

        self.state_dim=params['pxRes']
        self.img_pl = tf.placeholder(
            tf.float32, shape=[None,
                               self.state_dim,
                               self.state_dim, 1], name='input')

        if crop:
            image = tf.map_fn(lambda img: tf.random_crop(
                img,
                [self.params['pxRes'],
                 self.params['pxRes'],
                 1]), self.img_pl)
        else:
            image = self.img_pl
        print(image)

        # load graph and replace input
        # (changes input queue to simple feed dict)
        if os.path.isdir(params['classNN']):
            pathToGraph = tf.train.latest_checkpoint(params['classNN']) + \
                          ".meta"
            self.pathToModel = tf.train.latest_checkpoint(params['classNN'])
        elif  "model.ckpt" in params['classNN']:
            pathToGraph = params['classNN'] + ".meta"
            self.pathToModel = params['classNN']
        else:
            print("invalid path/classNN!")
            exit("-14")
        self.modelID = self.pathToModel.split("-")[-1]

        sc = "classConvNet"
        if "convNet" in params['classNN']:
                sc = "convNet"

        # load graph and replace input
        # (changes input queue to simple feed dict)
        self.saver = tf.train.import_meta_graph(
            pathToGraph,
            import_scope=sc,
            clear_devices=True,
            input_map={sc+'/inf/deflatteni1': image})
        self.nn = sess.graph.get_tensor_by_name(
            sc+"/"+sc+"/inf/out/BiasAdd:0")
        if self.params['veryveryverbose']:
            self.nn = tf.Print(self.nn, [self.nn], "outputClass:", first_n=20)


        self.isTraining = sess.graph.get_tensor_by_name(
            sc+"/isTraining_pl:0")
        self.keep_prob = sess.graph.get_tensor_by_name(
            sc+"/keep_prob_pl:0")

    def restoreWeights(self):
        # restore neural network weights
        # either from specific model
        # or from newest model in classNN
        print("restoring ... {}".format(self.pathToModel))
        self.saver.restore(self.sess, self.pathToModel)

    def runPrediction(self, inputs):
        # no sigmoid on output!!!
        print(self.nn, inputs.shape)
        return self.sess.run(self.nn, feed_dict={
            self.img_pl: inputs,
            self.isTraining: False,
            self.keep_prob: 1.0
        })


def loadData(params):
    if params['loadValBlob'] is not None:
        imgs, labels, files = ld.loadValBlob(params)
    elif params['validationData'] is not None:
        imgs, labels, files = ld.loadValidationData(params)
    else:
        raise RuntimeError("set either loadValBlob or validationData to supply validation data")
    print("Validation images shape", imgs.shape)
    print("Validation images min/max", imgs.min(), imgs.max())
    print("Validation images data type", imgs.dtype)

    return imgs, labels, files

if __name__ == "__main__":
    # load parameters from command line and config file
    params = parseNNArgs.parseArgs()
    params['type'] = "classifier"

    # set output directory
    if os.path.isdir(params['classNN']):
        out_dir = params['classNN']
    else:
        out_dir = os.path.dirname(params['classNN'])
    print("Results/Summaries/Logs will be written to: {}\n".format(out_dir))
    params['out_dir'] = out_dir

    # pipe log to file if not in interactive mode
    try:
        if os.environ['IS_INTERACTIVE'] != 'true':
            sys.stdout.flush()
            logFile = open(os.path.join(out_dir, "logEval"), 'a')
            sys.stdout = sys.stderr = logFile
    except KeyError:
        pass

    # start tensorflow session
    if params['noGPU']:
        tfconfig = tf.ConfigProto(
                device_count = {'GPU': 0}
        )
    else:
        tfconfig = None

    sess = tf.Session(config=tfconfig)

    # load data
    imgs, labels, files = loadData(params)

    # create tensorflow network
    nn = ClassConvNetEval(sess, params)

    # restore neural network weights
    nn.restoreWeights()

    # predict classes for validation data:
    mbs = params['miniBatchSize']
    #mbs = int(out_dir.split("/")[-1].split("_")[3])
    results = nn.runPrediction(imgs[0:mbs])
    print(imgs.shape, results.shape)
    correct = 0
    fn = 0
    fp = 0
    tp = 0
    tn = 0
    total = len(imgs)
    # for i in range(total):
    rng = int(len(imgs)/mbs)
    # rng = 3
    for idx in range(rng):
        bidx1 = (idx+1) * mbs
        bidx2 = min((idx+2) * mbs,total)
        # print(bidx1, bidx2, total)
        resT = nn.runPrediction(imgs[bidx1:bidx2])
        # print(resT.shape)
        results = np.append(results, resT)

    print(results.shape, total)

    for i in range(len(results)):
            # no sigmoid on output!!!
            if calcAcc.sigmoid(results[i]) < params['threshold']:
                if labels[i] == 0:
                    correct += 1
                    resCurrent = True
                    tn += 1
                else:
                    fn += 1
                    resCurrent = False
            else:
                if labels[i] == 1:
                    correct += 1
                    resCurrent = True
                    tp += 1
                else:
                    fp += 1
                    resCurrent = False
                    # print(files[i])
                    if params['veryverbose']:
                            print(resCurrent, calcAcc.sigmoid(results[i]),
                                  results[i], labels[i], files[i])

    calcAcc.calcAccuracy(params, None, results,
                         labels[:len(results)],
                         threshold=False)
    calcAcc.calcAccuracy(params, None, results,
                         labels[:len(results)],
                         threshold=params['threshold'])
    calcAcc.calcAccuracy(params, None, results,
                         labels[:len(results)],
                         threshold=0.9)
    np.save("results.npy", results)
    np.save("labels.npy", labels[:len(results)])
