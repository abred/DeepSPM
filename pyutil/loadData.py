import readImgs
import os
import pickle
import numpy as np

# code to load input data from disk
def loadData(out_dir, params):
    img_train, l_train, f_train, img_val, l_val, f_val = \
        readImgs.readData(params)

    print(img_train.shape, l_train.shape)
    with open(os.path.join(out_dir, "img_train.npy"), 'wb') as f:
        pickle.dump(img_train, f)
    with open(os.path.join(out_dir, "l_train.npy"), 'wb') as f:
        pickle.dump(l_train, f)
    with open(os.path.join(out_dir, "f_train.npy"), 'wb') as f:
        pickle.dump(f_train, f)

    for i in range(len(img_val)):
        print(img_val[i].shape, l_val[i].shape)
    with open(os.path.join(out_dir, "img_val.npy"), 'wb') as f:
        pickle.dump(img_val, f)
    with open(os.path.join(out_dir, "l_val.npy"), 'wb') as f:
        pickle.dump(l_val, f)
    with open(os.path.join(out_dir, "f_val.npy"), 'wb') as f:
        pickle.dump(f_val, f)

    return img_train, l_train, f_train, img_val, l_val, f_val

# code to read validation data from disk
# no need to split it
def loadValidationData(params):
    img_val, l_val, f_val = readImgs.readValidationData(params)
    out_dir = params['out_dir']

    print(img_val.shape, l_val.shape)

    return img_val, l_val, f_val

# code to restore precomputed validation data from disk
# (used to e.g. from a previous run)
def loadValBlob(params):
    out_dir = params['loadValBlob']
    with open(os.path.join(out_dir, "img_val.npy"), 'rb') as f:
        img_val = pickle.load(f)
    with open(os.path.join(out_dir, "l_val.npy"), 'rb') as f:
        l_val = pickle.load(f)
    with open(os.path.join(out_dir, "f_val.npy"), 'rb') as f:
        f_val = pickle.load(f)

    numGoodVal = 0
    for fn in f_val:
        if "good" in fn:
            numGoodVal += 1

    print("good val/total {} {} {}".format(
        float(numGoodVal)/float(len(img_val)), numGoodVal,
        len(img_val)))

    print(img_val.shape, l_val.shape, f_val.shape)

    return img_val, l_val, f_val


# code to restore input data from disk
# (used to e.g. resume a previous run)
def restoreData(out_dir):
    with open(os.path.join(out_dir, "img_train.npy"), 'rb') as f:
        img_train = pickle.load(f)
    with open(os.path.join(out_dir, "img_val.npy"), 'rb') as f:
        img_val = pickle.load(f)

    with open(os.path.join(out_dir, "l_train.npy"), 'rb') as f:
        l_train = pickle.load(f)
    with open(os.path.join(out_dir, "l_val.npy"), 'rb') as f:
        l_val = pickle.load(f)

    with open(os.path.join(out_dir, "f_train.npy"), 'rb') as f:
        f_train = pickle.load(f)
    with open(os.path.join(out_dir, "f_val.npy"), 'rb') as f:
        f_val = pickle.load(f)

    numGoodTrain = 0
    for fn in f_train:
        if "good" in fn:
            numGoodTrain += 1

    print("good train/total {} {} {}".format(
        float(numGoodTrain)/float(len(img_train)), numGoodTrain,
        len(img_train)))

    print(img_train.shape, l_train.shape, f_train.shape)

    return img_train, l_train, f_train, img_val, l_val, f_val
