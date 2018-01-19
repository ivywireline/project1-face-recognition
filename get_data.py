
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib

actors_file = "facescrub_actors.txt"
actresses_file = "facescrub_actresses.txt"

actors_list = list(set([a.split("\t")[0] for a in open(actors_file).readlines()]))
actresses_list = list(set([a.split("\t")[0] for a in open(actresses_file).readlines()]))





def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()


#Note: you need to create the uncropped folder first in order
#for this to work

for a in actors_list:
    name = a.split()[1].lower()
    i = 0
    for line in open(actors_file):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], "uncropped/actors/"+filename), {}, 45)
            if not os.path.isfile("uncropped/actors/"+filename):
                continue


            print filename
            i += 1

for a in actresses_list:
    name = a.split()[1].lower()
    i = 0
    for line in open(actresses_file):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], "uncropped/actresses/"+filename), {}, 45)
            if not os.path.isfile("uncropped/actresses/"+filename):
                continue


            print filename
            i += 1
