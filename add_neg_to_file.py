import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as keras
#import tensorflow.contrib.keras.datasets
#from tensorflow.contrib.keras.models import Sequential
#from tensorflow.contrib.keras.layers.core import Flatten, Dense, Dropout
#from tensorflow.contrib.keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
#from tensorflow.contrib.keras.optimizers import SGD
import cv2
import os
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import math
import h5py
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, ConstantScheme, SequentialScheme
from fuel.datasets.hdf5 import H5PYDataset
from fuel.converters.base import fill_hdf5_file, check_exists, progress_bar


def load_images_from_dir(data_dir, ext=".jpg"):
    imagesFiles = sorted([f for f in os.listdir(data_dir) if f.endswith(ext)])
    #imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(ext)]
    #imgs = [np.array(cv2.imread(os.path.join(data_dir, f), 0)) for f in imagesFiles]
    #imgs = [cv2.resize(x, size) for x in imgs]

    return imagesFiles

img_dir = "cutimg"
data = []
f_list = load_images_from_dir(img_dir, ext=".jpg")
for f in f_list:
    img = cv2.imread(os.path.join(img_dir, f))
    #np.transpose(img,(2,0,1))
    data.append(img)

#print len(data)
data = np.array(data)
data = np.transpose(data, (0,3,1,2))
#print data.shape
label = np.array([[10]*len(data)]).T
#print label.shape
#print label[0:2,:]
#train_neg = data[0:8000]
#test_neg = data[8000:]
train_len = int(len(data)*0.8)
test_len = len(data) - train_len

split_dict = {
     'train': {'features': (0, 73257),
               'targets': (0, 73257)},
    'test': {'features': (73257, 99289), 
              'targets': (73257, 99289)},
    'extra': {'features': (99289, 630420),
            'targets': (99289, 630420)},
    'train_neg': {'features': (630420, 630420+train_len),
            'targets': (630420, 630420+train_len)},
    'test_neg': {'features': (630420+train_len, 630420+len(data)),
            'targets': (630420+train_len, 630420+len(data))}
            }

with h5py.File('svhn_format_2.hdf5') as f:
    with h5py.File('new_more_neg_noresize_120kneg.hdf5', 'w') as nf:
        features = nf.create_dataset('features', (f['features'].shape[0]+data.shape[0], 3, 32, 32), dtype='uint8')
        targets = nf.create_dataset('targets', (f['features'].shape[0]+data.shape[0], 1), dtype='uint8')
        features[...] = np.vstack((f['features'], data))
        targets[...] = np.vstack((f['targets'], label))
        features.dims[0].label = 'batch'
        features.dims[1].label = 'channel'
        features.dims[2].label = 'height'
        features.dims[3].label = 'width'
        targets.dims[0].label = 'batch'
        targets.dims[1].label = 'index'
        nf.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    #f['features'].resize((f['features'].shape[0] + data.shape[0]), axis = 0)
    #f['features'][-data.shape[0]:] = data
    #f['targets'].resize((f['targets'].shape[0] + label.shape[0]), axis = 0)
    #f['targets'][-label.shape[0]:] = label
    #f.attrs['split'] = H5PYDataset.create_split_array(split_dict)



