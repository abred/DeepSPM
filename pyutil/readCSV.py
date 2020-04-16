import csv
import glob
from scipy import misc
import numpy as np

def readActions(fname):
    actionStrings = []
    actionForbiddenAreas = []
    actionMinReqSpots = []
    resetActions=[]
    measurementAction=-1
    with open(fname) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)
        count=0
        for row in reader:
            count+=1
            if count==1:
                continue
            actionStrings.append(str(row[0]))
            actionForbiddenAreas.append(float(row[1]))
            actionMinReqSpots.append(float(row[2]))
            if int(row[3])>0:
                resetActions.append(count-2)
            if int(row[4])>0:
                measurementAction = count-2
    return actionStrings, actionForbiddenAreas, actionMinReqSpots, resetActions, measurementAction
