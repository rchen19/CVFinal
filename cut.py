import numpy as np
#import tensorflow as tf
#import tensorflow.contrib.keras as keras
import cv2
import os
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import math
from random import randint, uniform
import h5py
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, ConstantScheme, SequentialScheme
from fuel.datasets.hdf5 import H5PYDataset
'''
train_neg = {}

train_set = H5PYDataset('svhn_format_1.hdf5', which_sets=('train',), subset=slice(0,2))
print "number of examples", train_set.num_examples
print train_set.sources
print train_set.provides_sources
print train_set.axis_labels

datastream=DataStream(train_set, iteration_scheme=SequentialScheme(examples=train_set.num_examples, batch_size=1))
i=0
for data in datastream.get_epoch_iterator():

    box_height, box_lable, box_left, box_top, box_width, features = data


train_neg = {}

'''

img_dir = "gs"

def load_images_from_dir(data_dir, ext=".jpg"):
    imagesFiles = sorted([f for f in os.listdir(data_dir) if f.endswith(ext)])
    #imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(ext)]
    #imgs = [np.array(cv2.imread(os.path.join(data_dir, f), 0)) for f in imagesFiles]
    #imgs = [cv2.resize(x, size) for x in imgs]

    return imagesFiles

def cut(f_list):

    #print len(f_list)
    #pick a random image
    f_num = randint(0, len(f_list)-1) #randint includes the higher bound
    #print f_num
    img = cv2.imread(os.path.join(img_dir, f_list[f_num]))
    
    #print img.shape
    #resize randomly
    #size_factor = uniform(300./min(img.shape[0:2]), 3.0)
    #print 50./min(img.shape[0:2])
    #print size_factor
    #img = cv2.resize(img, dsize=None, fx=size_factor, fy=size_factor)
    #rotate randomly

    #cut ransomly
    r = randint(0, img.shape[0]-32)
    c = randint(0, img.shape[1]-32)

    img_out = img[r:r+32, c:c+32,:]
    return img_out

if __name__ == "__main__":
    f_list = load_images_from_dir(img_dir, ext=".jpg")
    for i in xrange(40000):
        file_name = "{:05d}".format(i+80000)
        img_out = cut(f_list)
        cv2.imwrite(os.path.join("cutimg", "{}.jpg".format(file_name)), img_out)





