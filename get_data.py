
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

# You need to place the following text files in the directory the script is in
# for it to work

actors_file = "facescrub_actors.txt"
actresses_file = "facescrub_actresses.txt"

actors_list = list(set([a.split("\t")[0] for a in open(actors_file).readlines()]))
actresses_list = list(set([a.split("\t")[0] for a in open(actresses_file).readlines()]))

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.



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

#Note: you need to create the uncropped/actors and uncropped/actresses folders first in order
#for this to work
if not os.path.exists("uncropped"):
    os.makedirs("uncropped")

if not os.path.exists("cropped"):
    os.makedirs("cropped/actors")
    os.makedirs("cropped/actresses")

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
            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 45)
            if not os.path.isfile("uncropped/"+filename):
                continue
            try:
                # Get the coordinates
                x1 = int(line.split()[5].split(",")[0])
                y1 = int(line.split()[5].split(",")[1])
                x2 = int(line.split()[5].split(",")[2])
                y2 = int(line.split()[5].split(",")[3])
                # Gray scale, crop, then resize
                image_file = imread("uncropped/" + filename)
                gray = rgb2gray(image_file)
                cropped = gray[y1:y2, x1:x2]
                resized_image = imresize(cropped, (32, 32))
                imsave("cropped/actors/" + filename, resized_image, cmap = cm.gray)
            except Exception as e:
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
            timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 45)
            if not os.path.isfile("uncropped/"+filename):
                continue
            try:
                # Get the coordinates
                x1 = int(line.split()[5].split(",")[0])
                y1 = int(line.split()[5].split(",")[1])
                x2 = int(line.split()[5].split(",")[2])
                y2 = int(line.split()[5].split(",")[3])
                # Gray scale, crop, then resize
                image_file = imread("uncropped/" + filename)
                gray = rgb2gray(image_file)
                cropped = gray[y1:y2, x1:x2]
                resized_image = imresize(cropped, (32, 32))
                imsave("cropped/actresses/" + filename, resized_image, cmap = cm.gray)
            except Exception as e:
                continue

            print filename
            i += 1
