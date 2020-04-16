import glob
import numpy as np
import os

import readImgsUtil as ru

def parseNormFactor(paths, paramNF):
    if paramNF is not None:
        normFactor = paramNF
    else:
        normFactor = "1e-10"
    if ";" in paths and ";" not in normFactor:
        nf = normFactor
        for i in range(paths.count(";")):
            normFactor += ";" + nf
    return normFactor


# code to read real molecule images
def readData(params):
    print("reading real images")

    # validation or training mode?
    # if params['validationData'] is not None:
    #     paths = params['validationData']
    # else:
    paths = params['in_dir']

    normFactorTrain = parseNormFactor(paths, params['normFactorTrain'])

    pathsVal = []
    if params['valInDir1'] is not None:
        pathsVal.append(params['valInDir1'])
    if params['valInDir2'] is not None:
        pathsVal.append(params['valInDir2'])
    if params['valInDir3'] is not None:
        pathsVal.append(params['valInDir3'])
    if params['valInDir4'] is not None:
        pathsVal.append(params['valInDir4'])
    if params['valInDir5'] is not None:
        pathsVal.append(params['valInDir5'])
    if params['valInDir6'] is not None:
        pathsVal.append(params['valInDir6'])

    normFactorVal = []
    if params['valInDir1'] is not None:
        normFactorVal.append(
            parseNormFactor(params['valInDir1'],
                            params['normFactorVal1']))
    if params['valInDir2'] is not None:
        normFactorVal.append(
            parseNormFactor(params['valInDir2'],
                            params['normFactorVal2']))
    if params['valInDir3'] is not None:
        normFactorVal.append(
            parseNormFactor(params['valInDir3'],
                            params['normFactorVal3']))
    if params['valInDir4'] is not None:
        normFactorVal.append(
            parseNormFactor(params['valInDir4'],
                            params['normFactorVal4']))
    if params['valInDir5'] is not None:
        normFactorVal.append(
            parseNormFactor(params['valInDir5'],
                            params['normFactorVal5']))
    if params['valInDir6'] is not None:
        normFactorVal.append(
            parseNormFactor(params['valInDir6'],
                            params['normFactorVal6']))

    if params['allForTrain']:
        img_train, l_train, f_train = \
            readDataImpl(params, paths, normFactorTrain, "train")
        img_val = []
        l_val = []
        f_val = []
    else:
        img_train, l_train, f_train, img_val, l_val, f_val = \
            readDataImpl(params, paths, normFactorTrain, "train")
        img_val = np.array(img_val, dtype=np.float32)
        img_val.shape = (img_val.shape[0],
                          img_val.shape[1],
                          img_val.shape[2],
                          1)
        l_val = np.array(l_val, dtype=np.float32)
        l_val.shape = (l_val.shape[0], 1)
        f_val = np.array(f_val)
        img_val = [img_val]
        l_val = [l_val]
        f_val = [f_val]

    # just some reshaping
    img_train = np.array(img_train, dtype=np.float32)
    img_train.shape = (img_train.shape[0],
                        img_train.shape[1],
                        img_train.shape[2],
                        1)
    l_train = np.array(l_train, dtype=np.float32)
    l_train.shape = (l_train.shape[0], 1)
    f_train = np.array(f_train)

    for i in range(len(pathsVal)):
        it, lt, ft = \
            readDataImpl(params, pathsVal[i], normFactorVal[i], "val")
        it = np.array(it, dtype=np.float32)
        it.shape = (it.shape[0],
                    it.shape[1],
                    it.shape[2],
                    1)
        lt = np.array(lt, dtype=np.float32)
        lt.shape = (lt.shape[0], 1)
        ft = np.array(ft)
        img_val.append(it)
        l_val.append(lt)
        f_val.append(ft)

    return img_train, l_train, f_train, img_val, l_val, f_val


def readValidationData(params):
    print("reading real images for validation")
    paths = params['validationData']

    if False:
        # possible if no further preprocessing necessary
        img_val = []
        l_val = []
        f_val = []

        print(os.path.join(paths, "good", "*.npy"))
        fls = glob.glob(os.path.join(paths, "good", "*.npy"))
        for fl in fls:
            # print(fl)
            f = np.load(fl)
            img_val.append(f)
            l_val.append(1)
            f_val.append(fl)

        print(os.path.join(paths, "bad", "*.npy"))
        fls = glob.glob(os.path.join(paths, "bad", "*.npy"))
        for fl in fls:
            # print(fl)
            f = np.load(fl)
            img_val.append(f)
            l_val.append(0)
            f_val.append(fl)
    else:
        normFactor = parseNormFactor(paths, params['normFactorVal1'])
        img_val, l_val, f_val = \
            readDataImpl(params, paths, normFactor, "val")

    # just some reshaping
    img_val = np.array(img_val, dtype=np.float32)
    img_val.shape = (img_val.shape[0],
                      img_val.shape[1],
                      img_val.shape[2],
                      1)
    l_val = np.array(l_val, dtype=np.float32)
    l_val.shape = (l_val.shape[0], 1)
    f_val = np.array(f_val)

    return img_val, l_val, f_val

def readDataImpl(params, paths, normFactors, kind):
    print("reading images from", paths)
    if params['suffix'] is None:
        suffix = "Default"
    else:
        suffix = params['suffix']
    if params['RANSAC']:
        if suffix == "Default":
            suffix = "Ransac"
        else:
            suffix = "Ransac" + suffix
    print("suffix: {}".format(suffix))

    dirs = ["good", "bad"]
    normFactor = normFactors.split(";")
    files = []
    for idx, p in enumerate(paths.split(";")):
        print("path", p)
        for d in dirs:
            files += glob.glob(os.path.join(p, d, "*" + suffix + ".npy"))
        print("number of images (including sub-images)", len(files))
        print("found images", files[:5], "...")

    # split into training and validation sets
    # with both good and bad images in each set
    return ru.splitGoodBad(params, files, float(normFactor[0]), kind)
