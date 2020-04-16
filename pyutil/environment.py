import math
import ransacWrapper as rw
from findEmptySpot import *
from envClient import EnvClient

import png
import numpy as np
import sys
import time
import os
import tensorflow as tf
import readCSV
import normSTMImg
import scipy.ndimage.filters as spf
from scipy import ndimage
import glob
import datetime

def printT(s):
    sys.stdout.write(s + '\n')

class Environment(object):
    def __init__(self, sess, params, agent):

        self.params = params
        self.sess = sess
        self.agent = agent

        self.actions, self.actionForbiddenAreas, self.actionMinReqSpots ,self.resetActions , self.measurementAction= \
            readCSV.readActions(self.params['actionFile'])

        if self.params['threshold'] < 0 :
            self.randomTerm=True
        else:
            self.randomTerm=False

        self.env = EnvClient(sess, params, self.agent)

        self.numActions = len(self.actions)

        self.state = None
        self.currEp = None
        self.currStep = None
        self.currInEpStep = None

        self.pxRes = params['pxRes']
        self.size = params['sizeNano']

        self.cellSize=self.params['cellSize']

        self.currX = 0.0
        self.currY = 0.0

        self.lastScanX=self.currX
        self.lastScanY=self.currY

        self.ransac = rw.RansacWrapper(self.pxRes)

        self.zRange=None
        self.initApproachArea()
        open(os.path.join(self.params['out_dir'], "mask.npy"), 'ab')

        self.allowedActions = []
        self.meanScore=None

        open(os.path.join(self.params['out_dir'], "mask.npy"), "ab")

        self.clip = params['clip']
        if params['normFactorTrain'] is None:
            self.normFactor = 1e-10
        else:
            self.normFactor = params['normFactorTrain']

        self.debThresh = self.normFactor * params['debrisThreshold']

        imgsP=os.path.join(self.params['out_dir'], "imgsCollect/*raw.npy")
        self.collectImageNumber=len(glob.glob(imgsP))

        fn = os.path.join(self.params['out_dir'], "dataCollLog.txt")
        self.dataCollLog = open(fn, "a")

    def prepareAction(self):
        actPosYPx, actPosXPx, actFreeSzPx = findEmptySpotAbs(
            self.state,
            self.params['ThEmptyArea'] *10.0 ,
            self.imageNameString())
        if self.params['emptySpotDOG'] and \
           actFreeSzPx < self.params['emptySpotPlaneMinSz']:
            actPosYPx, actPosXPx, actFreeSzPx = findEmptySpotDOG(
                self.state,
                self.imageNameString())

        # Calculate absolute action position in nm
        # set tip position to this point
        self.currX = float(self.currX - self.size/2) + \
                     ( float(actPosXPx)/float(self.pxRes) * self.size)
        self.currY = float(self.currY - self.size/2) + \
                     (1 - (float(actPosYPx)/float(self.pxRes)) )* self.size


        freeSizeNano=actFreeSzPx*float(self.size)/float(self.pxRes)
        self.allowedActions = []

        for i in range(self.numActions):
            if self.actionMinReqSpots[i]<=freeSizeNano:
                # indicating action is allowed
                self.allowedActions.append(i)

    def initApproachArea(self):
        self.noDebresCounter=0
        self.getApproachArea()
        if self.params['restoreApproachAreaMask'] is not "" and \
           self.params['restoreApproachAreaMask'] is not None:
            printT("restoring approach area... {}".format(
                self.params['restoreApproachAreaMask']))
            self.mask = np.load(os.path.join(
                self.params['restoreApproachAreaMask']))
        else:
            printT("starting new approach area...")
            self.mask = np.zeros((2*self.locSz+1,2*self.locSz+1),dtype=np.bool)


        self.currX = 0.0
        self.currY = 0.0

        self.lastScanX=self.currX
        self.lastScanY=self.currY

        if self.params['spiral'] == "simple":
            self.computeLocationIDs()

    def getApproachArea(self):
        sz = self.env.getApproachArea()
        self.zRange = self.env.getZRange()
        printT("self.zRange: "+ str(self.zRange))
        #Subtracting safety margin
        self.maxY = sz/2 - self.params['margin']
        self.maxX = sz/2 - self.params['margin']
        self.locSz = int(self.maxX/self.cellSize)

    def switchApproachArea(self):
        printT("starting new approach area...")
        self.env.switchApproachArea()
        self.getApproachArea()
        self.mask = np.zeros((2*self.locSz+1,2*self.locSz+1),dtype=np.bool)
        self.noDebresCounter=0

        self.currX = 0.0
        self.currY = 0.0

        self.lastScanX=self.currX
        self.lastScanY=self.currY

        if self.params['spiral'] == "simple":
            self.computeLocationIDs()

#   Function does 3 things:
#       1. Execute Action
#       2. Mark forbidden area
#       3. Calculate Reward and Termination
#   Returns:
#       1. The new state
#       2. The reward
#       3. Is the new state terminal?
    def act(self, actID, ep,inEpStep, step, isEval=False):
        self.currEp = ep
        self.currStep = step
        self.currInEpStep = inEpStep


        if(self.randomTerm):
            self.params['threshold']= np.random.random_sample()

        if self.state is not None:
            # Here we excecute the action
            if self.params['veryverbose']:
                print('executing actID:',actID, self.actions[actID])
            self.env.act(self.actions[actID], self.currX, self.currY)


            self.markForbiddenArea(actID)


        if not actID == self.measurementAction:
            self.dataCollLog.write(str(datetime.datetime.now())+ "\t"+str(time.time())+ "\t" +"[tipshaping]"+"\t"+"NAN"+"\t"+"NAN" +
                                    "\t"+"NAN"+"\t"+str(self.lastScanX)+"\t"+str(self.lastScanY)+"\n")
            self.dataCollLog.flush()



        state=self.state
        # We record the new image
        # This also prepares the next action and sets 'self.allowedActions'
        nextState=self.getState()

        # compute reward/terminal
        terminalP = self.getTermNet(nextState)

        if self.params['reward'] == 'cl':
            reward = self.getClassifierReward(state, nextState, terminalP)
        elif self.params['reward'] == 'sCl':
            reward = self.getSoftClassifierReward(state, nextState, terminalP)
        elif self.params['reward'] == 'sClSuM':
            reward = self.getSoftClassifierRewardSubMean(state, nextState,
                                                         terminalP)
        elif self.params['reward'] == "stepF":
            reward = self.getStepFinalReward(state, nextState, terminalP)

        elif self.params['reward'] == 'step':
            reward = self.getStepReward(state, nextState, terminalP)

        terminal = self.isTerminal(isEval ,terminalP)


        if self.params['verbose']:
            printT("prob. for good probe: " + str(terminalP) +
                   " param[threshold]:" + str(self.params['threshold'])  +
                   " probe classified as good: "+ str(terminal))

        return nextState, reward, terminal, terminalP, np.copy(self.allowedActions)


    # Is the state terminal?
    def isTerminal(self, isEval ,terminalP= None):
        if (terminalP is None):
            if self.params['classNN'] is None:
                terminalP, reward = self.getReward()
            else:
                terminalP = self.getTermNet(self.state)
        terminal = (terminalP > self.params['threshold'] )
        return terminal

    #Mark forbidden area based on current position and provided radius in nm
    def markForbiddenAreaRad(self, forbiddenRadius):
        x = self.currX/self.cellSize
        y = self.currY/self.cellSize

        for i in range(-self.locSz,self.locSz+1):
            for j in range(-self.locSz,self.locSz+1):
                if self.mask[i+self.locSz,j+self.locSz] == True:
                    continue
                dist = np.sqrt((i-x)*(i-x) + (j-y)*(j-y))
                maxDist=(1.0)*forbiddenRadius/self.cellSize
                if dist<maxDist:
                    self.mask[i+self.locSz,j+self.locSz] = True
        if self.params['veryveryverbose']:
            fn = self.imageNameString() + "-mask.png"
            printT("storing mask to file: {}".format(fn))
            png.from_array(self.mask.astype(np.bool_), 'L').save(fn)
        os.unlink(os.path.join(self.params['out_dir'], "mask.npy"))
        np.save(os.path.join(self.params['out_dir'], "mask.npy"), self.mask)

    #Mark forbidden area resulting from action
    def markForbiddenArea(self, actID, multiplier=1.0):
        forbiddenRadius=multiplier*float(self.actionForbiddenAreas[actID])
        if self.params['veryveryverbose']:
            print("forbidden radius:"+str(forbiddenRadius))
        self.markForbiddenAreaRad(forbiddenRadius)


    def cleanTip(self):
        self.markForbiddenAreaRad(150)
        self.setNewTipPos()
        return self.env.cleanTip(self.currX,self.currY)


    def getState(self, noNewTipPos=False):

        if not noNewTipPos:
            self.setNewTipPos()

        if self.zRange is None:
            self.zRange = self.env.getZRange()

        noDebris=False
        self.noDebresCounter=0.0
        while not noDebris:
            self.noDebresCounter+=1.0
            contact=False
            while(not contact):
                if os.path.exists(os.path.join(self.params['out_dir'], "stop")):
                    self.agent.terminate()
                # We acquire the new image
                self.state = self.env.getState(self.currX, self.currY,
                                            self.size, self.pxRes)

                ctryCount=0
                while (np.max(self.state)-np.min(self.state) )<1e-25 and (np.mean(self.state)>0):
                    if os.path.exists(os.path.join(self.params['out_dir'], "stop")):
                        self.agent.terminate()
                    printT("CRASH DETECTED")
                    self.dataCollLog.write(str(datetime.datetime.now())+ "\t"+str(time.time())+ "\t" +"[crash]"+"\t"+"NAN"+"\t"+"NAN" +
                        "\t"+"NAN"+"\t"+str(self.currX)+"\t"+str(self.currY)+"\n")
                    self.dataCollLog.flush()
                    if ctryCount == 1:
                        self.agent.terminate()

                    if not self.cleanTip():
                        self.agent.terminate()

                    self.state = self.env.getState(self.currX, self.currY,
                                                self.size, self.pxRes)
                    ctryCount += 1


                mean=np.mean(self.state)
                maxDev=np.abs(mean-self.zRange)
                if maxDev > 3e-10:
                    contact=True
                else:
                    self.dataCollLog.write(str(datetime.datetime.now())+ "\t"+str(time.time())+ "\t" +"[noContact]"+"\t"+"NAN"+"\t"+"NAN" +
                        "\t"+"NAN"+"\t"+str(self.currX)+"\t"+str(self.currY)+"\n")
                    self.dataCollLog.flush()
                    printT("no Contact, new approach, maxDeviation:"+str(maxDev))
                    self.env.newApproach()


            noDebris = ( (np.max(self.state) - np.min(self.state) ) < self.debThresh ) or (self.noDebresCounter>250)
            if not noDebris:
                self.dataCollLog.write(str(datetime.datetime.now())+ "\t"+str(time.time())+ "\t" +"[debris]"+"\t"+"NAN"+"\t"+"NAN" +
                    "\t"+"NAN"+"\t"+str(self.currX)+"\t"+str(self.currY)+"\n")
                # We want to stall until we find an image without debris
                self.markForbiddenArea(actID=0, multiplier=(np.sqrt(float(self.noDebresCounter))*0.5) )

                self.dataCollLog.flush()

                self.setNewTipPos()



        if self.params['veryveryverbose']:
            stateOut=np.copy(self.state)
            stateOut[:]=(stateOut[:]+1) *128.0
            fn = self.imageNameString() + "-scan-raw.png"
            printT("storing file: {}".format(fn))
            png.from_array(stateOut.astype(np.uint8), 'L').save(fn)

        self.scanRaw=self.state
        if self.params['veryveryverbose']:
            fn = self.imageNameString() +"-scan-raw.npy"
            np.save(fn, self.state)

        # use ransac to fit a plane to current state and substract it
        if self.params['RANSAC']:
            self.state = self.ransac.apply(self.state)

        self.state = normSTMImg.normImg1(self.state,
                                         fac=self.normFactor,
                                         clip=self.clip)

        if self.params['veryveryverbose']:
            stateOut=np.copy(self.state)
            stateOut[:]=(stateOut[:]+1) *128.0
            fn = self.imageNameString() + "-scan-processed.png"
            printT("storing file: {}".format(fn))
            png.from_array(stateOut.astype(np.uint8), 'L').save(fn)

        self.scanProcessed=self.state
        if self.params['veryveryverbose']:
            fn = self.imageNameString() + "-scan-processed.npy"
            np.save(fn, self.state)

        # we try to find a good spot to perform the next action
        # and determine which actions will be allowed
        self.lastScanX=self.currX
        self.lastScanY=self.currY
        self.prepareAction()

        # Adjust the shape to allow the image to be processed by the CNN
        self.state.shape = (1, self.state.shape[0], self.state.shape[1], 1)
        return self.state

    # Reset the tip. In other words, drastically change it.
    # THIS IS CHANGED FOR NORMAL OPERATION MODE
    def reset(self, ep, isEval=False, globActStep=None):
        self.currEp = ep
        self.currInEpStep = 0
        if globActStep is not None:
            self.currStep=globActStep

        if self.params['maxBadImgCount'] is not None:
            return self.collectSamples( ep, isEval)




        # self.state is only None in the beginning
        # We do not want to destroy the tip in the beginning except when self.params['initDestroy'] is set
        if (self.state is not None) or self.params['initDestroy']:
            for i in range(5):
                actID = np.random.choice(self.resetActions)
                if self.params['verbose']:
                    printT("resetting with action:"+str(actID))
                ns_, r_, term, tP, aa_ = self.act( actID, self.currEp, self.currInEpStep, self.currStep, isEval)
                if tP < 0.1:
                    break

        if self.state is None:
            self.getState()

        terminal = self.isTerminal(isEval)
        return self.state, self.allowedActions, terminal




    def collectSamples(self, ep, isEval=False):

        # self.state is only None in the beginning
        # We do not want to destroy the tip in the beginning except when self.params['initDestroy'] is set
        badImgCounter=0
        self.agent.stopLearning=True # disable learning thread
        self.dataCollLog.write(str(datetime.datetime.now())+ "\t"+str(time.time())+ "\t" +"[stopTipshaping]"+"\t"+str(self.collectImageNumber)+"\t"+str(badImgCounter) +
                                           "\t"+"NAN"+"\t"+"NAN"+"\t"+"NAN"+"\n")
        self.dataCollLog.flush()

        while True:
                actID = self.measurementAction #We do not want to change the tip, so we stall

                self.currX=self.lastScanX
                self.currY=self.lastScanY

                ns_, r_, term, tP, aa_ = self.act( actID, self.currEp, self.currInEpStep, self.currStep, isEval)
                bad= tP < self.params['threshold']

                self.collectImageNumber+=1
                fn = self.imageNameString(not bad) +"-scan-raw.npy"
                np.save(fn, self.scanRaw)
                fn = self.imageNameString(not bad) +"-scan-processed.npy"
                np.save(fn, self.scanProcessed)

                if bad:
                    badImgCounter+=1
                    self.dataCollLog.write(str(datetime.datetime.now())+ "\t"+str(time.time())+ "\t" +"[collectingBadSample]"+"\t"+str(self.collectImageNumber)+"\t"+str(badImgCounter) +
                                           "\t"+str(tP)+"\t"+str(self.lastScanX)+"\t"+str(self.lastScanY)+"\n")
                    self.dataCollLog.flush()

                else:
                    badImgCounter=0
                    self.dataCollLog.write(str(datetime.datetime.now())+ "\t"+str(time.time())+ "\t" +"[collectingValidSample]"+"\t"+str(self.collectImageNumber)+"\t"+str(badImgCounter) +
                                           "\t"+str(tP)+"\t"+str(self.lastScanX)+"\t"+str(self.lastScanY)+"\n")
                    self.dataCollLog.flush()


                if badImgCounter > self.params['maxBadImgCount']:
                    self.dataCollLog.write(str(datetime.datetime.now())+ "\t"+str(time.time())+ "\t" +"[startTipshaping]"+"\t"+str(self.collectImageNumber)+"\t"+str(badImgCounter) +
                                           "\t"+"NAN"+"\t"+"NAN"+"\t"+"NAN"+"\n")
                    self.dataCollLog.flush()
                    break

        self.agent.stopLearning=False # enable learning thread
        terminal = False
        return self.state, self.allowedActions, terminal

    # used in simple spiral tip positioning
    def computeLocationIDs(self):
        self.ids = []
        sh = self.size//2
        d = self.size+sh
        while True:
            if leg == 0:
                x += 1
                if x == layer:
                    leg += 1
            elif leg == 1:
                y += 1
                if y == layer:
                    leg += 1
            elif leg == 2:
                x -= 1
                if -x == layer:
                    leg += 1
            elif leg == 3:
                y -= 1
                if -y == layer:
                    leg = 0
                    layer += 1

            if ((abs(x))*d+sh) > self.maxX or ((abs(y))*d+sh) > self.maxY:
                break
            self.ids.append((x,y))

    def setNewTipPos(self):
        if self.params['spiral'] == "simple":
            self.currX, self.currY = self.getNextInSpiral()
        else:
            self.currX, self.currY = self.getNextGoodClosest()

        # approach area full, switching ...
        if self.currX == None:
            self.switchApproachArea()
        if self.params['veryveryverbose']:
            printT("next tip pos: "+ str(self.currX)+ ", "+str(self.currY) )

    def getNextInSpiral(self):
        x = None
        y = None
        for idx in self.ids:
            x, y = idx[0], idx[1]
            xid = x
            yid = y
            if self.mask[xid, yid] == True:
                continue
        return x*self.size*1.5, y*self.size*1.5

    def getNextGoodClosest(self):
        # Tip position in mask scale
        x = float(self.currX)/float(self.cellSize)
        y = float(self.currY)/float(self.cellSize)
        maxMovement=self.params['maxMovement']/float(self.cellSize)
        regTerm=self.params['spiralReg']
        mn = self.locSz*100
        mnD = self.locSz*100
        xm=0
        ym=0
        found=False

        for i in range(-self.locSz,self.locSz+1):
            for j in range(-self.locSz,self.locSz+1):
                if self.mask[i+self.locSz,j+self.locSz] == True:
                    continue

                ds2 = np.sqrt(float((i-x)*(i-x))+ float((j-y)*(j-y)))
                if (ds2>maxMovement):
                    continue

                ds1 = max(abs(i-0), abs(j-0))
                dist=(regTerm*ds1+ds2)
                if dist<mn or (not found):
                    found=True
                    mn=dist
                    xm=i
                    ym=j

        if not found:
            return None, None
        # Return tip position in nm
        return xm*self.cellSize, ym*self.cellSize

    def imageNameString(self, collectGood=None):
        if collectGood is not None:
            if collectGood:
                    result= os.path.join(self.params['out_dir'], "imgsCollect",
                        "Img_" + str(self.collectImageNumber) +
                        "-Step_" + str(self.currStep) +
                        "-Ep_" + str(self.currEp) +
                        "-InEpStep_" + str(self.currInEpStep)+ "-good" )
            else:
                    result= os.path.join(self.params['out_dir'], "imgsCollect",
                        "Img_" + str(self.collectImageNumber) +
                        "-Step_" + str(self.currStep) +
                        "-Ep_" + str(self.currEp) +
                        "-InEpStep_" + str(self.currInEpStep) +"-bad" )

        else:
            result= os.path.join(self.params['out_dir'], "imgs",
                "Img_" + str(self.collectImageNumber) +
                "-Step_" + str(self.currStep) +
                "-Ep_" + str(self.currEp) +
                "-InEpStep_" + str(self.currInEpStep) )

        return result

#=========================================================================
#                            REWARD FUNCTIONS
#=========================================================================
    def getTermNet(self, state):
        threshold = self.params['threshold']
        res = self.rewardClassNet.runPrediction(state)

        # Calculate the termination probability
        terminal = 1.0/(1.0+np.exp(-float(res)))
        return terminal

        # get reward using Classifier
    def getClassifierReward(self, state, nextState, terminal):
        if terminal > self.params['threshold']:
            reward= self.params['rewardFinal']
            return reward

        resOld = self.rewardClassNet.runPrediction(state)

        # Calculate the termination probability
        terminalOld = 1.0/(1.0+np.exp(-float(resOld)))

        if terminalOld < terminal:
            reward = self.params['rewardPos']
        else:
            reward = self.params['rewardNeg']

        return reward


    # get reward using soft Classifier sub mean
    def getSoftClassifierRewardSubMean(self, state, nextState, terminal):
        resOld = self.rewardClassNet.runPrediction(state)

        resNew= -np.log((1.0/terminal)-1.0)

        if(self.meanScore is None):
            self.meanScore=resOld
        else:
            self.meanScore=self.meanScore*0.99+resOld*0.01

        resOld-=self.meanScore
        resNew-=self.meanScore

        terminalNew = 1.0/(1.0+np.exp(-float(resNew)   ))
        terminalOld = 1.0/(1.0+np.exp(-float(resOld)   ))

        if terminal > self.params['threshold']:
            reward= (1.0-terminalOld)* self.params['rewardFinal'] + \
                    float(self.params['rewardPos'])
            return reward

        reward= (terminalNew-terminalOld)* float(self.params['rewardFinal']) +\
                float(self.params['rewardPos'])
        return reward


    # get reward using soft Classifier
    def getSoftClassifierReward(self, state, nextState, terminal):
        resOld = self.rewardClassNet.runPrediction(state)

        # Calculate the termination probability
        terminalOld = 1.0/(1.0+np.exp(-float(resOld)))

        if terminal > self.params['threshold']:
            reward= (1.0-terminalOld)* self.params['rewardFinal'] + \
                    float(self.params['rewardPos'])
            return reward

        reward= (terminal-terminalOld)* float(self.params['rewardFinal']) + \
                float(self.params['rewardPos'])
        return reward


    # get simple step reward
    def getStepReward(self, state, nextState, terminal):
        return self.params['rewardPos']

    def getStepFinalReward(self, state, nextState, terminal):
        if terminal > self.params['threshold']:
            reward= self.params['rewardFinal']
            return reward
        else:
            reward= self.params['rewardPos']
            return reward
