import tensorflow as tf
import struct
import numpy as np
import os
import sys
import time
import socket
import png

import parseNNArgs
import outDir
import readCSV
import time
import glob

# parsing code for test stm-server
def parseElem(strng, verbose=False):
    if verbose:
        print("parseElem", strng)
    if strng[-1].isdigit():
        return [float(strng), '']
    else:
        return [float(strng[:-1]), strng[-1]]


def parseStr(strng, verbose=False):
    if verbose:
        print("parseStr", strng)
    strng = strng.split(",")
    if "scan" in strng[0]:
        posX, posXsuf = parseElem(strng[0].split("(")[-1])
        posY, posYsuf = parseElem(strng[1])
        sz, szSuf = parseElem(strng[2])
        px, _ = parseElem(strng[3].split(")")[0])
        if verbose:
            print("scanparam", posX, posXsuf, posY, posYsuf, sz, szSuf, px)

        return ["scan", posX, posY, sz, px]
    elif "tipshaping" in strng[0]:
        posX, posXsuf = parseElem(strng[0].split("(")[-1])
        posY, posYsuf = parseElem(strng[1])
        dip, dipSuf = parseElem(strng[2])
        bias, biasSuf = parseElem(strng[3])
        timing, timingSuf = parseElem(strng[4].split(")")[0])
        if verbose:
            print("tipshaping", posX, posXsuf, posY, posYsuf, dip, dipSuf,
                  bias, biasSuf, timing, timingSuf)

        return ["tipshaping", posX, posY, dip, bias, timing]
    elif "tipclean" in strng[0]:
        posX, posXsuf = parseElem(strng[0].split("(")[-1])
        posY, posYsuf = parseElem(strng[1].split(")")[0])
        if verbose:
            print("tipclean", posX, posXsuf, posY, posYsuf)
        return ["tipclean", posX, posY]
    elif "getparam" in strng[0]:
        if verbose:
            print("getparam")
        return ["getparam"]
    elif "getReward" in strng[0]:
        if verbose:
            print("getReward")
        return ["getReward"]
    elif "reset" in strng[0]:
        isEval = bool(strng[0].split("(")[-1].split(")")[0])
        if verbose:
            print("resetting tip {}".format(isEval))
        return ["reset", isEval]
    elif "movearea" in strng[0]:
        dire = strng[0].split("(")[-1][:-1]
        if verbose:
            print("change approach area in direction {}".format(dire))
        return ["movearea", dire]

# simulates stm-server, for testing
class EnvServer:
    def __init__(self, sess, params):
        self.params = params
        self.sess = sess
        self.host = "localhost"
        self.port = self.params['port']
        if not self.params['dummyServer']:
            print("You have to set the dummyServer flag! Exiting.")
            exit(-1)
        else:
            self.sz = 800
            self.approachArea = \
                np.random.rand(self.sz, self.sz).astype(np.float32)

        self.actions, self.actionForbiddenAreas, self.actionMinReqSpots, _, _ = readCSV.readActions(self.params['actionFile'])
        self.isolated=False
        self.filenames= glob.glob(self.params['dummyImageDir'])


    def parseRequest(self, request):
        request = request.decode('utf-8')
        #if self.params['veryverybose']:
        print("=======================Request:" + request)
        request = request.replace(" ", "")
        args = parseStr(request, verbose=self.params['veryverbose'])
        if self.params['veryverbose']:
            print(args)
        if args[0] == "scan":
            return self.processScan(args)
        elif args[0] == "tipshaping":
            return self.processAction(args,request)
        elif args[0] == "getparam":
            return self.processGetParam(args)
        elif args[0] == "movearea":
            return self.processSwitchArea(args)
        elif args[0] == "getReward":
            return self.processGetReward(args)
        elif args[0] == "reset":
            return self.processReset(args)
        elif args[0] == "tipclean":
            return self.processTipClean(args)
        return "invalid".encode('utf-8')

    def processTipClean(self, args):
        posX = float(args[1])
        posY = float(args[2])
        #self.isolated=False
        print("FIXING ISOLATION PROBLEM")
        return "1".encode('utf-8')


    def processScan(self, args):
        posX = int(args[1])
        posY = int(args[2])
        sz = int(args[3])
        # px = args[4]
        # posX = 300
        # posY = 300
        px = int(args[4])
        pxH = int(px/2)
        posY += self.sz//2
        posX += self.sz//2

        fn=self.filenames[np.random.randint(0,len(self.filenames))]
        print(fn)
        img=np.load(fn).astype(np.float32)
        img.shape=(64,64)
        response = img
            #np.random.randn(pxH*2,pxH*2).astype(np.float32)*1e-10
        print("MAX, MIN: ",np.max(response),np.min(response))
#            np.copy(self.approachArea[0:pxH*2,0:pxH*2])

        if self.isolated:
            print("TIP IS ISOLATED")
            response[:]=1.0

        response = response.byteswap()  #swap bytes for transmission
        time.sleep(1)
        meta = struct.pack('>ii', px, px)
        return meta + response.tostring()



    def processAction(self, args, request):
        return str(np.random.rand()).encode('utf-8')

    def processGetParam(self, args):
        self.sz = 800
        response = "Range:"+str(self.sz*1e-9)
        response = response.encode('utf-8')
        return response

    # This is not present when working with a real STM
    def processGetReward(self, args):
        rew=-np.random.randint(100)
        return str(rew).encode('utf-8')

    def processSwitchArea(self, args):
        crsh = np.random.randint(2)
        response = "Approach area changed with " + str(crsh) + " crashes"
        response = response.encode('utf-8')
        return response

    def processReset(self, args):
        return str(0).encode('utf-8')

    def run(self):
        if self.params['dummyServer']:
            print("Starting Dummy Server without simulated environment.")
        else:
            print("Starting Server with simulated environment.")

        listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_socket.bind((self.host, self.port))
        listen_socket.listen(1)
        print('Serving on port {} ...'.format(self.port))

        client_connection, client_address = listen_socket.accept()


        while True:
            request = client_connection.recv(1024)
            response = self.parseRequest(request)
            client_connection.sendall(response)



def main(params):
    tfconfig = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
    with tf.Session(config=tfconfig) as sess:
        server = EnvServer(sess, params)
        server.run()
    print("server dying...")

if __name__ == "__main__":
    params = parseNNArgs.parseArgs()
    out_dir = outDir.setOutDir(params)
    params['out_dir'] = out_dir
    main(params)
