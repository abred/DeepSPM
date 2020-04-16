import sys
import os
import json
import getopt

# parse command line arguments
def parseArgs():
    params = parse(sys.argv[1:])

    try:
        if os.environ['IS_INTERACTIVE'] == 'true':
            params['version'] = 'tmp'
    except KeyError:
        pass

    print(params, params['version'])

    return params

def parse(argv):
    params = dict()

#___________________________________________________________
#   General Parameters for Classifier and Agent
#___________________________________________________________

    # root dir for output
    params['root'] = None

    # maximum number of training steps
    params['numTrainSteps'] = 1000

    # size of mini batch
    params['miniBatchSize'] = 16

    # specify a seed for random number generator, uses random seed if None
    params['seed'] = None

    # a version string , for reference
    params['version'] = '1'

    # use dropout in networks
    params['dropout'] = False

# DATA PRE-PROCESSING
    # Randomly flips images horizontally and vertically
    params['distorted'] = False

    params['distortBrightnessRelative'] = 0.0
    params['distortContrast'] = 0.0
    params['distortGaussian'] = 0.0
    params['distortSaltPepper'] = 0.0

    # normalization factor used to scale images before they are fed into networks
    # applies to training data for classifier and to images recorded by agent
    params['normFactorTrain'] = None

# TRAINING PARAMETERS
    # which optimizer to use: 'sgd', 'momentum', or 'adam'
    params['optimizer'] = 'momentum'

    params['learning-rate'] = 0.1
    params['lr-decay'] = False
    params['momentum'] = 0.9
    params['mom-decay'] = False
    params['weight-decay'] = 0.0

    # threshold for classification into good or bad probe
    params['threshold'] = 0.5

    # there will be a pooling operation after each of the first X levels of the cnn
    params['numPool'] = None

    # parameter for batchnorm
    params['batchnorm-decay'] = 0.9

    # size of images to be processed by the networks in pixels
    params['pxRes'] = 64

    # path to a classifier network you want to load
    params['classNN'] = None

    # controls how much info is printed
    params['verbose'] = False
    params['veryverbose'] = False
    params['veryveryverbose'] = False

    # use CPU only, very slow
    params['noGPU'] = False

# ___________________________________________________________
#   Parameters for the Classifier
# ___________________________________________________________
    # path to training data
    params['in_dir'] = None

    # don't split the dataset into train and val, the validation set is specified
    # separately using --valInDirX (and normFactorValX)
    params['allForTrain'] = False

    # paths to validation datasets
    params['valInDir1'] = None
    params['valInDir2'] = None
    params['valInDir3'] = None
    params['valInDir4'] = None
    params['valInDir5'] = None
    params['valInDir6'] = None

    # factors used to normalize the validation datasets
    params['normFactorVal1'] = None
    params['normFactorVal2'] = None
    params['normFactorVal3'] = None
    params['normFactorVal4'] = None
    params['normFactorVal5'] = None
    params['normFactorVal6'] = None

    # raw input data is supplied as one image per file
    # for faster resuming/evaluation it is, after loading it the first time,
    # stored as one numpy binary blob.
    # you can copy these blobs to a separate location and use them as input
    # for subsequent experiments by setting this parameter to the path
    # to the containing directory
    params['loadBlob'] = None

    # use batchnorm in classifier network
    params['batchnorm'] = False

    # defines the vgg-style network,
    # for the cnn part specifies the number of feature maps in each level,
    # each convolution is repeated three times
    # for the fully connected part specifies the number of neurons
    params['cnnSetSizes'] = "32,64,96"
    params['fcSizes'] = "2048,2048"

    # extra factor to penalize false positives harder
    params['penalizeFP'] = False
    # extra factor to increase the weight of positive/good samples in the loss
    params['relWeightPosSamples'] = None

    # path to validation data used by the evaluation scripts
    # by setting validationData the files are loaded from scratch
    # by setting loadValBlob precomputed numpy binary blobs are used
    # (analogue to loadBlob (see above))
    params['validationData'] = None
    params['loadValBlob'] = None

    # preload all data into gpu memory
    # (recommendation: True (no augmentation if set to False))
    params['preload'] = True

    # use 'area under the curve' instead of cross entropy EXPERIMENTAL
    params['aucLoss'] = False

    # read only image file names that end in this suffix
    # if 'None' this is set to the string "Default"
    params['suffix'] = None

# ___________________________________________________________
#   Parameters for the Agent
# ___________________________________________________________

# CONNECTION TO SERVER
    # the address of the SPM server you want to connect to
    params['host'] = "localhost"
    # the port at which you want to connect
    params['port'] = 50008

    # We start a dummy server pretending to be an STM.
    # This will automatically activate the 'dummyServer' option.
    params['startServer'] = False

    # A dummy server will give random syntactically correct answers.
    # automatically activated by 'startServer'
    params['dummyServer'] = False

    # use together with 'startServer' and 'dummyServer'
    # the dummy server will read images in this directory and return
    # them when asked to scan an area
    params['dummyImageDir'] = None

# SELECTION OF NEXT SCANNING REGION
    # determines how the next scanning region is selected
    # 'closest' will find a compromise between staying close to the center and the last region
    # 'simple' will use a fixed spiral
    params['spiral'] = "closest" # "simple"

    # regularization factor for selecting the next region
    # large value leads to more regular path
    params['spiralReg'] = 1.0

    # Maximum allowed movement between to images in nm.
    # If the required movement is larger, the approach area is changed
    params['maxMovement'] = 500.0

    # Size of the gridcells used for motion planing in nm
    params['cellSize'] = 10.0

    # size of scanning region in nanometer
    params['sizeNano'] = 10

    # threshold for the difference between max and min values in the image.
    # If this is exceeded, the region is considered to contain excessive roughness.
    params['debrisThreshold'] = 6

    # Safety margin to subtract from approach area in nm
    params['margin'] = 0.0

# PROBE CONDITIONING
    # path to .csv describing possible conditioning actions
    params['actionFile'] = None

    # Threshold for finding an empty area in nm
    # determines how close a pixel value has to be to the plane to be considered empty
    params['ThEmptyArea'] = 0.01

    # Use difference of Gaussians blob detector to find empty spot. EXPERIMENTAL
    params['emptySpotDOG'] = False
    # A parameter for when using 'emptySpotDOG'
    params['emptySpotPlaneMinSz'] = 0

    # Should we reset (destroy) the tip at the beginning?
    params['initDestroy'] = False


# CNN PARAMETERS
    # defines the vgg-style network,
    # for the cnn part specifies the number of feature maps in each level,
    # each convolution is repeated three times
    # for the fully connected part specifies the number of neurons
    params['cnnSetSizesDQN'] = "32,64,96"
    params['fcSizesDQN'] = "2048,2048"

    # use batchnorm in agent network
    params['batchnormDQN'] = False

    # Smartly initialize output neurons
    params['outputInitialization'] = False


# REINFORCEMENT LEARNING
    # hard or soft update of target dqn
    params['noHardResetDQN'] = False

    # factor for soft update
    params['tau'] = None

    # frequency of hard update
    params['resetFreq'] = None

    # RL discount factor
    params['gamma'] = 0.99

    # get [q]values of target network for actions where main network is max
    # (recommendation: True)
    params['doubleDQN'] = False

    # use DQN with the dueling network extension
    # (recommendation: False)
    params['duelingDQN'] = False

    # Should target network update (batchnorm) moving_mean/variance using tau
    # or use batch statistics
    params['updateTargetBNStatsWithTau'] = False

    # choose a reward scheme
    # 'stepF': 'rewardPos' after each step, 'rewardFinal' when episode is finished
    # 'step': 'rewardPos' after each step
    # 'cl': 'rewardPos' if classifier output improved, 'rewardNeg' else, 'rewardFinal' when episode is finished
    # 'sCl': soft version of 'cl'
    # 'sClSuM': another soft version of 'cl'
    params['reward'] = None
    params['rewardPos'] = None
    params['rewardNeg'] = None
    params['rewardFinal'] = None

# EXPLORATION AND LEARNING

    # the number of conseq. times a bad probe has to be detected by classifier,
    # before a probe shaping episode is initiated
    # IMPORTANT: set this to 'None' for training the agent.
    # if 'None' the probe will be reset (destroyed)  after each successful episode.
    # Set it to small positive number during deepSPM operation.
    params['maxBadImgCount'] = None

    # number of initial completely random episodes
    params['randomEps'] = 100

    # start learning if at least x elements in buffer
    params['startLearning'] = 200

    # max number of steps in episode
    params['stepsTillTerm'] = 100

    # anneal epsilon linearly in X steps
    params['annealSteps'] = 20000

    # start linear decay of epsilon at
    params['epsilonStart'] = 0.95

    # stop linear decay at this; epsilon is constant afterwards
    params['epsilonStop'] = 0.0

    # only learn, no exploration
    params['onlyLearn'] = False

    # size of replay buffer
    params['replaySz'] = None

    # stop if buffer full
    params['termAtFull'] = False
    
    # maximum number of probe conditioning episodes
    params['numEpisodes'] = 1000

    # use Huber loss instead of mean square error
    params['huberLoss'] = False

    # this option is for evaluating the performance of the agent
    # if activated, it will alternate between using the agent and a random selection of tip shaping actions
    params['interEval'] = False

    # EXPERIMENTAL
    # Use these two only together and together with 'interEval'.
    # This will alternate between a 'fixedAction' (provide index)
    # and 'agentB' (provide path)
    params['fixedAction'] = 0
    params['agentB'] = None

    # perform pure evaluation episodes every 25 episodes
    # DEPRECATED
    params['evaluation'] = False


# LOADING AND STORING
    # path to an action network we want to load
    params['dqnNN'] = None

    # use x network for agent init
    params['useClassNN'] = False

    # load old replay buffer
    params['loadReplay'] = None

    # load an old model
    params['loadModel'] = None

    # periodically store tensorflow action network model
    params['storeModel'] = False

    # periodically store replay buffer
    params['storeBuffer'] = False

    params['restoreApproachAreaMask'] = None

    # we are resuming a run that was previoulsy stoped
    params['resume'] = False

    # keep only newest n saved dqn models to save disc space; delete older ones
    params['keepNewestModels'] = 200

    # learning/exploration is performed in different threads, RECOmMENDED
    params['async'] = False

    # Make the exploration thread ssleep from time to time to simulated a slow microscope
    params['sleep'] = False

    # use with 'sleep'. specify a time one iteration should take
    params['sleepA'] = None

    # We pause exploring when the buffer is larger than this value, DEPRECATED
    params['limitExploring'] = None

# IMAGE PRE-PROCESSING
    # Apply RANSAC plane fitting and subtraction
    params['RANSAC'] = True

    #clip value to [-1.5,1.5] after normalization and plane subtraction
    params['clip'] = True

    # each experience is inserted 16 times into buffer
    # (rotating and mirroring)
    params['fullAugmentation'] = False


    try:
        print(argv)
        opts, args = getopt.getopt(argv, "i:",
                                   ['seed=',
                                    'resume=',
                                    'root=',
                                    'version=',
                                    'miniBatchSize=',
                                    'numEpisodes=',
                                    'numTrainSteps=',
                                    'distorted',
                                    'distortBrightnessRelative=',
                                    'distortContrast=',
                                    'distortGaussian=',
                                    'distortSaltPepper=',
                                    'dropout=',
                                    'learning-rate=',
                                    'lr-decay',
                                    'momentum=',
                                    'mom-decay',
                                    'optimizer=',
                                    'penalizeFP',
                                    'relWeightPosSamples=',
                                    'weight-decay=',
                                    'noPreload',
                                    'suffix=',
                                    'batchnorm',
                                    'batchnormDQN',
                                    'batchnorm-decay=',
                                    'threshold=',
                                    'cnnSetSizes=',
                                    'fcSizes=',
                                    'cnnSetSizesDQN=',
                                    'fcSizesDQN=',
                                    'numPool=',
                                    'allForTrain',
                                    'valInDir1=',
                                    'valInDir2=',
                                    'valInDir3=',
                                    'valInDir4=',
                                    'normFactorTrain=',
                                    'normFactorVal1=',
                                    'normFactorVal2=',
                                    'normFactorVal3=',
                                    'normFactorVal4=',
                                    'validationData=',
                                    'doubleDQN',
                                    'duelingDQN',
                                    'noHardResetDQN',
                                    'tau=',
                                    'resetFreq=',
                                    'loadLevel=',
                                    'reward=',
                                    'rewardPos=',
                                    'rewardNeg=',
                                    'rewardFinal=',
                                    'gamma=',
                                    'randomEps=',
                                    'stepsTillTerm=',
                                    'startLearning=',
                                    'async',
                                    'sleep',
                                    'sleepA=',
                                    'limitExploring=',
                                    'annealSteps=',
                                    'classNN=',
                                    'useClassNN',
                                    'dqnNN=',
                                    'blob=',
                                    'valBlob=',
                                    'replaySz=',
                                    'termAtFull',
                                    'loadReplay=',
                                    'loadModel=',
                                    'storeModel',
                                    'model=',
                                    'storeBuffer',
                                    'onlyLearn',
                                    'sizeNano=',
                                    'pxRes=',
                                    'evaluation',
                                    'verbose',
                                    'veryverbose',
                                    'veryveryverbose',
                                    'noGPU',
                                    'startServer',
                                    'host=',
                                    'port=',
                                    'spiral=',
                                    'restoreApproachAreaMask=',
                                    'actionFile=',
                                    'noRANSAC',
                                    'noClip',
                                    'emptySpotDOG',
                                    'emptySpotPlaneMinSz=',
                                    'updateTargetBNStatsWithTau',
                                    'fullAugmentation',
                                    'outputInitialization',
                                    'epsilonStart=',
                                    'epsilonStop=',
                                    'huberLoss',
                                    'aucLoss',
                                    'keepNewestModels=',
                                    'interEval',
                                    'fixedAction=',
                                    'agentB=',
                                    'maxBadImgCount=',
                                    'dummyServer',
                                    'debrisThreshold=',
                                    'dummyImageDir=',
                                   ])
        print(opts, args)
    except getopt.GetoptError as err:
        print('args parse error')
        print('args: ', argv)
        print(err)
        exit()
    for opt, arg in opts:
        print("+++++++")
        print(opt, arg)
        if opt == '--resume':
            params['resume'] = bool(int(arg))

        elif opt == '--root':
            params['root'] = arg

        elif opt == '--seed':
            params['seed'] = int(arg)

        elif opt == '--version':
            params['version'] = arg

        elif opt == '--numTrainSteps':
            params['numTrainSteps'] = int(arg)

        elif opt == '--numEpisodes':
            params['numEpisodes'] = int(arg)

        elif opt == '--miniBatchSize':
            params['miniBatchSize'] = int(arg)

        elif opt == '--noPreload':
            params['preload'] = False

        elif opt == '--distorted':
            params['distorted'] = True
        elif opt == '--distortBrightnessRelative':
            params['distortBrightnessRelative'] = float(arg)
        elif opt == '--distortContrast':
            params['distortContrast'] = float(arg)
        elif opt == '--distortGaussian':
            params['distortGaussian'] = float(arg)
        elif opt == '--distortSaltPepper':
            params['distortSaltPepper'] = float(arg)

        elif opt == '--dropout':
            params['dropout'] = float(arg)

        elif opt == '-i':
            params['in_dir'] = arg

        elif opt == '--valInDir1':
            params['valInDir1'] = arg

        elif opt == '--valInDir2':
            params['valInDir2'] = arg

        elif opt == '--valInDir3':
            params['valInDir3'] = arg

        elif opt == '--valInDir4':
            params['valInDir4'] = arg

        elif opt == '--normFactorTrain':
            params['normFactorTrain'] = arg

        elif opt == '--normFactorVal1':
            params['normFactorVal1'] = arg

        elif opt == '--normFactorVal2':
            params['normFactorVal2'] = arg

        elif opt == '--normFactorVal3':
            params['normFactorVal3'] = arg

        elif opt == '--normFactorVal4':
            params['normFactorVal4'] = arg

        elif opt == '--suffix':
            params['suffix'] = arg

        elif opt == '--batchnorm':
            params['batchnorm'] = True

        elif opt == '--batchnormDQN':
            params['batchnormDQN'] = True

        elif opt == '--batchnorm-decay':
            params['batchnorm-decay'] = float(arg)

        elif opt == '--threshold':
            params['threshold'] = float(arg)

        elif opt == '--cnnSetSizes':
            params['cnnSetSizes'] = arg

        elif opt == '--fcSizes':
            params['fcSizes'] = arg

        elif opt == '--cnnSetSizesDQN':
            params['cnnSetSizesDQN'] = arg

        elif opt == '--fcSizesDQN':
            params['fcSizesDQN'] = arg

        elif opt == '--numPool':
            params['numPool'] = int(arg)

        elif opt == '--allForTrain':
            params['allForTrain'] = True

        elif opt == '--learning-rate':
            params['learning-rate'] = float(arg)

        elif opt == '--lr-decay':
            params['lr-decay'] = True

        elif opt == '--momentum':
            params['momentum'] = float(arg)

        elif opt == '--mom-decay':
            params['mom-decay'] = True

        elif opt == '--weight-decay':
            params['weight-decay'] = float(arg)

        elif opt == '--optimizer':
            params['optimizer'] = arg

        elif opt == '--penalizeFP':
            params['penalizeFP'] = True

        elif opt == '--relWeightPosSamples':
            params['relWeightPosSamples'] = float(arg)

        elif opt == '--doubleDQN':
            params['doubleDQN'] = True

        elif opt == '--duelingDQN':
            params['duelingDQN'] = True

        elif opt == '--noHardResetDQN':
            params['noHardResetDQN'] = True

        elif opt == '--tau':
            params['tau'] = float(arg)

        elif opt == '--resetFreq':
            params['resetFreq'] = int(arg)

        elif opt == '--validationData':
            params['validationData'] = arg

        elif opt == '--reward':
            params['reward'] = arg

        elif opt == '--rewardPos':
            params['rewardPos'] = float(arg)

        elif opt == '--rewardNeg':
            params['rewardNeg'] = float(arg)

        elif opt == '--rewardFinal':
            params['rewardFinal'] = float(arg)

        elif opt == '--gamma':
            params['gamma'] = float(arg)

        elif opt == '--randomEps':
            params['randomEps'] = int(arg)

        elif opt == '--stepsTillTerm':
            params['stepsTillTerm'] = int(arg)

        elif opt == '--startLearning':
            params['startLearning'] = int(arg)

        elif opt == '--async':
            params['async'] = True

        elif opt == '--sleepA':
            params['sleepA'] = float(arg)

        elif opt == '--sleep':
            params['sleep'] = True

        elif opt == '--limitExploring':
            params['limitExploring'] = int(arg)

        elif opt == '--annealSteps':
            params['annealSteps'] = float(arg)

        elif opt == '--classNN':
            params['classNN'] = arg

        elif opt == '--useClassNN':
            params['useClassNN'] = True

        elif opt == '--dqnNN':
            params['dqnNN'] = arg

        elif opt == '--blob':
            params['loadBlob'] = arg

        elif opt == '--valBlob':
            params['loadValBlob'] = arg

        elif opt == '--replaySz':
            params['replaySz'] = int(arg)

        elif opt == '--termAtFull':
            params['termAtFull'] = True

        elif opt == '--loadReplay':
            params['loadReplay'] = arg

        elif opt == '--loadModel':
            params['loadModel'] = arg

        elif opt == '--storeModel':
            params['storeModel'] = True

        elif opt == '--storeBuffer':
            params['storeBuffer'] = True

        elif opt == '--onlyLearn':
            params['onlyLearn'] = True

        elif opt == '--sizeNano':
            params['sizeNano'] = int(arg)

        elif opt == '--pxRes':
            params['pxRes'] = int(arg)

        elif opt == '--evaluation':
            params['evaluation'] = True

        elif opt == '--verbose':
            params['verbose'] = True

        elif opt == '--veryverbose':
            params['veryverbose'] = True

        elif opt == '--veryveryverbose':
            params['veryveryverbose'] = True

        elif opt == '--noGPU':
            params['noGPU'] = True

        elif opt == '--startServer':
            params['startServer'] = True
            params['dummyServer']= True

        elif opt == '--host':
            params['host'] = arg

        elif opt == '--port':
            params['port'] = int(arg)

        elif opt == '--spiral':
            params['spiral'] = arg

        elif opt == '--restoreApproachAreaMask':
            params['restoreApproachAreaMask'] = arg

        elif opt == '--actionFile':
            params['actionFile']= (arg)

        elif opt == '--noRANSAC':
            params['RANSAC'] = False

        elif opt == '--noClip':
            params['clip'] = False

        elif opt == '--emptySpotDOG':
            params['emptySpotDOG'] = True

        elif opt == '--emptySpotPlaneMinSz':
            params['emptySpotPlaneMinSz'] = int(arg)

        elif opt == '--updateTargetBNStatsWithTau':
            params['updateTargetBNStatsWithTau'] = True

        elif opt == '--fullAugmentation':
            params['fullAugmentation'] = True

        elif opt == '--outputInitialization':
            params['outputInitialization'] = True

        elif opt == '--epsilonStart':
            params['epsilonStart'] = float(arg)

        elif opt == '--epsilonStop':
            params['epsilonStop'] = float(arg)

        elif opt == '--huberLoss':
            params['huberLoss'] = True

        elif opt == '--aucLoss':
            params['aucLoss'] = True

        elif opt == '--keepNewestModels':
            params['keepNewestModels'] = int(arg)

        elif opt == '--interEval':
            params['interEval'] = True
            params['initDestroy'] = True
            params['dummyServer'] = True

        elif opt == '--fixedAction':
            params['fixedAction'] = int(arg)

        elif opt == '--agentB':
            params['agentB'] = arg

        elif opt == '--maxBadImgCount':
            params['maxBadImgCount'] = int(arg)

        elif opt == '--dummyServer':
            params['dummyServer']= True

        elif opt == '--debrisThreshold':
            params['debrisThreshold'] = float(arg)

        elif opt == '--dummyImageDir':
            params['dummyImageDir'] = arg




    if params['veryveryverbose'] == True:
        params['veryverbose'] = True
        params['verbose'] = True
    elif params['veryverbose'] == True:
        params['verbose'] = True

    return params
