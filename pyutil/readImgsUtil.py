import ransacWrapper as rw
import normSTMImg as ni

import scipy.ndimage
import sys
import random

import numpy as np
from sklearn.utils import shuffle

def splitGoodBad(params, files, normFactor, kind):
    files = list(files)
    if params['allForTrain'] or kind == "val":
        sfCnt = int(len(files) * 1.0)
    else:
        sfCnt = int(len(files) * 0.7)

    if params['seed'] is None:
        seed = random.randrange(sys.maxsize)
    else:
        seed = params['seed']
    print("Seed was:", seed)
    random.seed(seed)

    train_files, val_files = splitFiles(files, params, sfCnt, kind)

    img_train = []
    img_val = []
    l_train = []
    l_val = []
    f_train = []
    f_val = []

    try:
        print(files[0])
    except:
        raise RuntimeError("no image files found")

    sz = np.load(files[0]).shape[0]
    ransac = rw.RansacWrapper(sz)

    print("loading train files", len(train_files))
    cnt = 0
    for fl in train_files:
        if params['verbose']:
            print(cnt, fl, normFactor)
        cnt += 1
        loadImg(fl, params, normFactor,
                img_train, l_train, f_train, ransac)

    if kind == "val":
        print("loading val files", len(train_files))
    elif not params['allForTrain']:
        print("loading val files", len(val_files))
    cnt = 0
    for fl in val_files:
        if params['verbose']:
            print(cnt, fl, normFactor)
        cnt += 1
        loadImg(fl, params, normFactor,
                img_val, l_val, f_val, ransac)

    if params['allForTrain'] or kind == "val":
        return img_train, l_train, f_train
    else:
        return img_train, l_train, f_train, img_val, l_val, f_val


# repeat until approx even split of good and bad images
# in training and val sets is achieved
# first x image go into training set, rest into val set
# shuffle after each try if uneven split
def splitFiles(files, params, sfCnt, kind):
    while True:
        rnd = random.randint(0, 2**31)
        print("random_state: {}".format(rnd))
        files = shuffle(files, random_state=rnd)

        train_files = files[:sfCnt]
        val_files = files[sfCnt:]
        numGoodTrain = 0
        numBadTrain = 0
        numGoodVal = 0
        numBadVal = 0

        for fl in train_files:
            if "good" in fl:
                numGoodTrain += 1
            else:
                numBadTrain += 1
        for fl in val_files:
            if "good" in fl:
                numGoodVal += 1
            else:
                numBadVal += 1

        if params['allForTrain'] or kind == "val":
            break

        fracGoodTotal = float(numGoodTrain+numGoodVal)/float(len(files))
        fracGoodTrain = float(numGoodTrain)/float(sfCnt)
        fracGoodVal  = float(numGoodVal)/float(len(files)-sfCnt)
        print("fraction good images total: {}".format(fracGoodTotal))
        print("fraction good images in training set: {}".format(fracGoodTrain))
        print("fraction good images in validation  set: {}".format(fracGoodVal))
        sys.stdout.flush()
        plusDeviation =  fracGoodTotal * 1.1
        negDeviation =  fracGoodTotal * 0.9
        if negDeviation < fracGoodTrain < plusDeviation and \
           negDeviation < fracGoodVal < plusDeviation:
            break
    return train_files, val_files

def loadImg(fl, params, normFactor, imgs, ls, fs, ransac):
    nearestNeighbour = False

    img = np.load(fl)
    if img.shape[1] == 128:
        if nearestNeighbour:
            img = img[::2, ::2]
        else:
            img = scipy.ndimage.zoom(img, (0.5, 0.5))
    elif img.shape[1] == 102:
        print("shouldnt exist, check data set", img.shape)
        img = scipy.ndimage.zoom(img, (64.0/102.0, 64.0/102.0))

    if params['RANSAC'] and not "Ransac" in fl:
        img = doRansac(img, ransac)
    clip = params['clip'] and not params['distortContrast']
    img = normImg(img, normFactor, clip=clip)

    imgs.append(img)
    if "good" in fl:
        ls.append(1)
    else:
        ls.append(0)
    fs.append(fl)

def doRansac(img, ransac):
    while True:
        try:
            img = ransac.apply(img)
            break
        except:
            pass
    return img

def normImg(img, normFactor, clip=True):
    if normFactor>0.0:
        img = ni.normImg1(img, fac=normFactor, clip=clip)
    return img
