import time
import os

# for organization, encode parameters in dir name
def setOutDir(params):
    timestamp = str(int(time.time()))
    try:
        jobid = os.environ['SLURM_JOBID']
    except:
        jobid = 'NOID'

    if params['root'] is None:
        root = os.path.join(os.environ['HOME'], "STM", "experiments")
    else:
        root = params['root']

    if os.environ['IS_INTERACTIVE'] == 'true':
        vers = "tmp"
    else:
        vers = params['version']

    out_dir = os.path.abspath(os.path.join(
        root,
        params.get("type", "unknown_type"), # classifier/agent
        "runs",
        vers,
        jobid + "_" + timestamp))

    if params.get("type", "unknown_type") == "agent":
        out_dir += "_mbs" + str(params['miniBatchSize'])
        out_dir += "_tau" + str(params['tau'])
        out_dir += "_es" + str(params['epsilonStart'])
        out_dir += "_en" + str(params['epsilonStop'])
        out_dir += "_opt" + str(params['optimizer'])
        out_dir += "_lr" + str(params['learning-rate'])
        if params['batchnorm']:
            out_dir += "_bn"
        else:
            out_dir += "_noBn"
        out_dir += "_rew" + str(params['reward'])

        print("Number of random episodes: ", params['randomEps'])
        out_dir += "_randEps" + str(params['randomEps'])

        print("gamma: ", params['gamma'])
        out_dir += "_gamma" + str(params['gamma'])

        if params['dqnNN'] is not None:
            try:
                out_dir += "_init" + params['dqnNN'].split("/")[8].split("_")[0]
            except:
                pass
        elif params['useClassNN'] is not None:
            try:
                out_dir += "_init" + params['classNN'].split("/")[8].split("_")[0]
            except:
                pass
        else:
            out_dir += "_initRand"
        return out_dir

    if params.get("type", "unknown_type") == "classifier":
        print("Number of training steps: ", params['numTrainSteps'])
        out_dir += "_" + str(params['numTrainSteps'])

    print("miniBatchSize: ", params['miniBatchSize'])
    out_dir += "_" + str(params['miniBatchSize'])

    print("dropout", params['dropout'])
    if params['dropout']:
        out_dir += "_" + "drp" + str(params['dropout'])
    else:
        out_dir += "_" + "noDrp"

    if params['distortBrightnessRelative'] or params['distortContrast']:
        params['distorted'] = True
    print("distorted", params['distorted'])
    if params['distorted']:
        out_dir += "_" + "augm"
        delta = params['distortBrightnessRelative']
        factor = params['distortContrast']
        stddev = params['distortGaussian']
        fracSP = params['distortSaltPepper']
        if delta != 0:
            out_dir += "_Br" + str(delta)
        if factor != 0:
            out_dir += "_Cntr-" + str(factor)
        if stddev != 0:
            out_dir += "_Gau-" + str(stddev)
        if fracSP != 0:
            out_dir += "_SP-" + str(fracSP)
    else:
        out_dir += "_" + "noAugm"

    print("batchnorm", params['batchnorm'])
    if params['batchnorm']:
        out_dir += "_" + "bn-" + str(params['batchnorm-decay'])
    else:
        out_dir += "_" + "noBn"



    out_dir += "_" + "cSz" + str(params['pxRes'])

    print("weight decay", params['weight-decay'])
    out_dir += "_wd" + str(params['weight-decay'])

    print("learning rate", params['learning-rate'])
    out_dir += "_lr" + str(params['learning-rate'])
    if params["lr-decay"]:
        out_dir += "Dc"

    print("momentum", params['momentum'])
    out_dir += "_mom" + str(params['momentum'])

    print("optimizer", params['optimizer'])
    out_dir += "_opt" + params['optimizer']

    if params['in_dir'] is not None:
        print("reading data from: ", params['in_dir'])
        out_dir += params['in_dir'].split("/")[-1]

    if params['aucLoss']:
        out_dir += "_auc"

    if params['penalizeFP']:
        out_dir += "_penFP"

    if params['relWeightPosSamples'] is not None:
        out_dir += "_wPos" + str(params['relWeightPosSamples'])

    if params['RANSAC']:
        out_dir += "_ransac"
    else:
        out_dir += "_noRansac"

    return out_dir
