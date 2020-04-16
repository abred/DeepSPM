import png
import numpy as np
import sys
import time
import os
import struct
import socket
import select
import random
import tensorflow as tf
import time
import datetime

"""
encapsulate network communication, client side
"""

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def empty_socket(sock):
    """remove the data present on the socket"""
    input = [sock]
    while 1:
        inputready, o, e = select.select(input,[],[], 0.0)
        if len(inputready)==0: break
        for s in inputready: s.recv(1)


class EnvClient:
    def __init__(self, sess, params, agent):
        self.agent=agent
        self.terminateOnFail=False
        self.params = params
        self.sess = sess
        if self.params['host'] is not None:
            self.host = self.params['host']
        else:
            self.host = "localhost"
        if self.params['port'] is not None:
            self.port = self.params['port']
        else:
            self.port = 50008
        print(self.host, self.port)
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.connect((self.host, self.port))
        self.tmout = 180.0
        self.soc.settimeout(self.tmout)
        self.lastSwitchTime=int(time.time())
        self.requestSendTime=time.time()

        fn = os.path.join(self.params['out_dir'], "clientLog.txt")
        self.logFile = open(fn, "a")

    def act(self, action, px, py):
        """ request to server/stm to perform specified action"""
        print("action:" + str(action))
        request = "tipshaping(" + \
                  str(float(px)) + "n," + \
                  str(float(py)) + "n," + \
                  action + ")"
        if action == "stall":
            if self.params['veryverbose']:
                print("======================Stalling action; not sending command to server======================")
                print(request)
            return
        sys.stdout.flush()

        while True:
            try:
                empty_socket(self.soc)
                self.sendRequest(request)
                response = self.soc.recv(30).decode('utf-8')
                break
            except Exception as e:
                print(e)
                if self.terminateOnFail:
                    self.agent.terminate()
                else:
                    self.terminateOnFail=True
                print("tipshaping failed, repeating...")
                print("Logging...")
                self.agent.logStuff()
                print("Logging...Done!")
                self.soc.settimeout(0)
                empty_socket(self.soc)
                self.soc.settimeout(self.tmout)
                sys.stdout.flush()
        if self.params['veryverbose']:
            print("response act:", response)

        print("res:"+ str(response))
        self.logResponse(str(response))

    def getApproachArea(self):
        """request (square) size of approach area"""
        request = "getparam(Range)"
        sys.stdout.flush()
        while True:
            try:
                empty_socket(self.soc)
                self.sendRequest(request)
                response = self.soc.recv(30).decode('utf-8')
                break
            except Exception as e:
                print(e)
                if self.terminateOnFail:
                    self.agent.terminate()
                else:
                    self.terminateOnFail=True
                print("getApproachArea failed, repeating...")
                print("Logging...")
                self.agent.logStuff()
                print("Logging...Done!")
                self.soc.settimeout(0)
                empty_socket(self.soc)
                self.soc.settimeout(self.tmout)
                sys.stdout.flush()

        if self.params['veryverbose']:
            print("response getapproacharea:", response)
        sz = float(response.split(':')[1])
        sz *= 1e9
        if self.params['veryverbose']:
            print("getparam response: {}".format(sz))
        self.logResponse(response)
        return sz

    def getZRange(self):
        """request zRange of stm (in meters)"""
        request = "getparam(zRange)"
        sys.stdout.flush()
        while True:
            try:
                empty_socket(self.soc)
                self.sendRequest(request)
                response = self.soc.recv(30).decode('utf-8')
                break
            except Exception as e:
                print(e)
                if self.terminateOnFail:
                    self.agent.terminate()
                else:
                    self.terminateOnFail=True
                print("getZRange failed, repeating...")
                print("Logging...")
                self.agent.logStuff()
                print("Logging...Done!")
                self.soc.settimeout(0)
                empty_socket(self.soc)
                self.soc.settimeout(self.tmout)
                sys.stdout.flush()

        if self.params['veryverbose']:
            print("response getZRange:", response)
        sz = float(response.split(':')[1])

        if self.params['veryverbose']:
            print("getparam getZRange response: {}".format(sz))
        self.logResponse(response)
        return -sz/2.0

    def getState(self, currX, currY, size, pxRes):
        """request a new recording/image"""
        request = "scan(" + str(currX) + "n," + str(currY) + \
                  "n," + str(size) + "n ," + str(pxRes) + ")"

        sys.stdout.flush()
        while True:
            try:
                empty_socket(self.soc)
                self.sendRequest(request)
                response = self.soc.recv(2*4)
                if self.params['veryverbose']:
                    print("response getstate shape:", response)
                szY, szX = struct.unpack('>ii', response)
                if self.params['veryverbose']:
                    print("decoded shape:", szY, szX)
                if pxRes != szY or pxRes != szX:
                    print("invalid size!")
                    self.sendExit()
                    exit(-6)

                response = recvall(self.soc, szY*szX*4)
                break
            except Exception as e:
                print(e)
                if self.terminateOnFail:
                    self.agent.terminate()
                else:
                    self.terminateOnFail=True
                print("getState failed, repeating...")
                print("Logging...")
                self.agent.logStuff()
                print("Logging...Done!")
                sys.stdout.flush()
                self.soc.settimeout(0)
                empty_socket(self.soc)
                self.soc.settimeout(self.tmout)

        img = np.fromstring(response, dtype=np.float32)
        img = img.byteswap()
        if self.params['veryverbose']:
            print("response getstate img:", img, img.shape)
            print("current image: ({}, {})".format(szY, szX))

        img.shape = (szY, szX)
        print(img.shape, np.min(img),np.max(img))

        if self.params['veryverbose']:
            print("image distance dff: {}".format(np.max(img) - np.min(img)))
        sys.stdout.flush()

        img = np.copy(img)
        self.logResponse("IMAGE")
        return img

    def newApproach(self):
        """request reapproach of probe to sample"""
        request = "approach(f)"
        currTime = int(time.time())
        self.lastSwitchTime = currTime
        sys.stdout.flush()
        while True:
            try:
                empty_socket(self.soc)
                self.sendRequest(request)
                response = self.soc.recv(64).decode('utf-8')
                self.soc.settimeout(self.tmout)
                break
            except:
                if self.terminateOnFail:
                    self.agent.terminate()
                else:
                    self.terminateOnFail=True
                print("newApproach failed, repeating...")
                print("Logging...")
                self.agent.logStuff()
                print("Logging...Done!")
                self.soc.settimeout(0)
                empty_socket(self.soc)
                self.soc.settimeout(self.tmout)
                sys.stdout.flush()


        print("response newApproach:", response)
        self.logResponse(response)

    def switchApproachArea(self):
        """request to move probe to a new approach area"""
        request = "movearea(y+)"
        currTime = int(time.time())
        self.lastSwitchTime = currTime
        sys.stdout.flush()
        while True:
            try:
                empty_socket(self.soc)
                self.sendRequest(request)
                self.soc.settimeout(60*30.0)
                response = self.soc.recv(64).decode('utf-8')
                self.soc.settimeout(self.tmout)
                break
            except:
                if self.terminateOnFail:
                    self.agent.terminate()
                else:
                    self.terminateOnFail=True
                print("movearea failed, repeating...")
                print("Logging...")
                self.agent.logStuff()
                print("Logging...Done!")
                self.soc.settimeout(0)
                empty_socket(self.soc)
                self.soc.settimeout(self.tmout)
                sys.stdout.flush()

        print("response movearea:", response)
        crsh = int(response.split('with ')[1].split(" crash")[0])
        print("switch area, number of crashes: {}".format(crsh))
        self.logResponse(response)

    def sendRequest(self, request):
        self.logRequest(request)
        if self.params['veryverbose']:
            print("request: {}, {}".format(request, len(request)))
        request = request.encode('utf-8')
        if self.params['veryverbose']:
            print(self.soc.getsockname())
        sys.stdout.flush()
        self.soc.send(request)


    def cleanTip(self, px, py):
        """request stm to clean its tip, only used in otherwise
        unrecoverable situations
        """

        request = "tipclean(" + str(float(px)) + "n," + str(float(py)) + "n)"
        sys.stdout.flush()

        while True:
            try:
                empty_socket(self.soc)
                self.sendRequest(request)
                self.soc.settimeout(60*30.0)
                response = self.soc.recv(30).decode('utf-8')
                self.soc.settimeout(self.tmout)
                break
            except Exception as e:
                print(e)
                if self.terminateOnFail:
                    self.agent.terminate()
                else:
                    self.terminateOnFail=True
                print("tip cleaning failed, repeating...")
                print("Logging...")
                self.agent.logStuff()
                print("Logging...Done!")
                self.soc.settimeout(0)
                empty_socket(self.soc)
                self.soc.settimeout(self.tmout)
                sys.stdout.flush()
        if self.params['veryverbose']:
            print("response tipclean:", response)
        if int(response)==1:
            return True
        else:
            return False
        self.logResponse(response)


    def logRequest(self, request):
        self.logFile.write(str(datetime.datetime.now())+ "\t"+
                           str(time.time())+ "\t >> "+request+ "\n")
        self.requestSendTime = time.time()
        self.logFile.flush()

    def logResponse(self, response):
        self.terminateOnFail=False
        responseTime = time.time()- self.requestSendTime
        self.logFile.write(str(datetime.datetime.now())+ "\t"+
                           str(time.time())+ "\t << "+response+ "\t"+
                           str(responseTime)+"\n")
        self.logFile.flush()
