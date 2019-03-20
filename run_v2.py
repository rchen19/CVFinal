'''
this script creates sliding window
on an input image, use trained model
to classify what is in the window
11 classes inlcuding 10 digits and non-digit
'''
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
#import matplotlib.pyplot as plt
import pandas as pd
import math
import h5py
#from fuel.streams import DataStream
#from fuel.schemes import ShuffledScheme, SequentialScheme, ShuffledExampleScheme
#from fuel.datasets.hdf5 import H5PYDataset
from model import *

def nms(corners, window, prob, threshold=0.2): #non-max suppression
    if not corners:
        return []
    elif len(corners) ==0 :
        return []
    else:
        pick = []
        corners_array = np.array(corners)
        window_array = np.array(window)
        prob_array = np.array(prob)
        if corners_array.dtype.kind == "i":
            corners_array = corners_array.astype("float")
        x1 = corners_array[:,0]
        y1 = corners_array[:,1]
        x2 = x1 + window_array
        y2 = y1 + window_array

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        indices = np.argsort(prob_array)
        #print "prob_array:", prob_array
        #print "indices:", indices

        while len(indices) > 0:
            last =len(indices) - 1
            i = indices[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[indices[:last]])
            yy1 = np.maximum(y1[i], y1[indices[:last]])
            xx2 = np.minimum(x2[i], x2[indices[:last]])
            yy2 = np.minimum(y2[i], y2[indices[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = overlap = (w * h) / area[indices[:last]]

            indices = np.delete(indices, np.concatenate(([last],
                            np.where(overlap > threshold)[0])))
        #print "pick:", pick
        return pick



def detect(image, model, probability_threshold=0.99999):
    image_out = image.copy()
    image = image.astype("float")
    #center the pixel values by subtract the mean
    image = image - image.mean(axis=(0,1,2), keepdims=True)
    h, w, _ = image.shape
    #image = cv2.resize(image, (120, 60))
    #h, w, _ = image.shape
    #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #check if size of image is larger than 24 x 24:
    if h<32 or w<32:
        raise Exception("Image size too small")


    corners_all = []
    predictions_all = []
    whichdigit_all = []
    window_all = []
    all_scales = []
    prob_all = []
    #best_prob = 0
    #scale_final = 1.
    #create image pyramid
    #for scale in [0.6, 0.8, 1., 1.2, 1.4]:
    for scale in [0.6, 0.8, 1.]:
        image_temp = cv2.resize(image, dsize=None, fx=scale, fy=scale)
        h, w, _ = image_temp.shape
        #create sliding windows
        corners = []
        windows = []
        for i in range(0,w-32,2):
            for j in range(0,h-32,2):
                window = image_temp[j:j+32, i:i+32, :]
                windows.append(window)
                corners.append((i,j))
        preds = model.predict(np.array(windows))

        digitcorners = []
        predictions = []
        whichdigit = []
        window_sizes = []
        prob = []
        for i in range(preds.shape[0]):
            if preds[i].max() > probability_threshold and preds[i].argmax()!=10:
                #print preds[i].argmax()
                #print preds[i]
                digitcorners.append((corners[i][0]/scale, corners[i][1]/scale))
                predictions.append(preds[i])
                whichdigit.append(preds[i].argmax())
                prob.append(preds[i].max())
                window_sizes.append(32./scale)
        corners_all += digitcorners
        predictions_all += predictions
        whichdigit_all += whichdigit
        window_all += window_sizes
        prob_all += prob
        #if predictions:
        #    if np.array(predictions).max() > best_prob:
        #        best_prob = np.array(predictions).max()
        #        corners_final = digitcorners
        #        predictions_final = predictions
        #        whichdigit_final = whichdigit
        #        scale_final = scale

    #scale corners coord back
    #corners_final_scaled_back = [(int(corner[0]/scale_final), int(corner[1]/scale_final)) for corner in corners_final]


    rect_color = (255, 0, 0)
    rect_thickness = 1
    pick = nms(corners_all, window_all, prob_all, threshold=0.35)
    corners_final = [corners_all[i] for i in pick]
    predictions_final = [predictions_all[i] for i in pick]
    whichdigit_final = [whichdigit_all[i] for i in pick]
    window_final = [window_all[i] for i in pick]

    #print "scale: {}".format(scale_final)

    print "number of digits found: {}".format(len(pick))
    for i in range(len(pick)):
        
        top_left = (int(corners_final[i][0]), int(corners_final[i][1]))
        window_size = window_final[i]
        print whichdigit_final[i]
        print top_left
        print predictions_final[i]
        bottom_right = (top_left[0]+int(window_size), top_left[1]+int(window_size))
        digit = whichdigit_final[i]
        cv2.rectangle(image_out, top_left, bottom_right, rect_color, rect_thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_out, str(digit), (top_left[0]-10, top_left[1]), font, 0.4, (0, 0, 0), 1, cv2.CV_AA)
    #cv2.imwrite("graded_images/{}_model{}_3.png".format(filename, model_num), image_out)
    #cv2.imwrite(os.path.join(INPUT_DIR, "{}_model{}_3.png".format(filename, model_num)), image_out)
    #cv2.imwrite(os.path.join(INPUT_DIR, "{}.png".format(filename)), image_out)
    return image_out

def load_images_from_dir(data_dir, ext=".jpg"):
    imagesFiles = sorted([f for f in os.listdir(data_dir) if f.endswith(ext)])
    #imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(ext)]
    #imgs = [np.array(cv2.imread(os.path.join(data_dir, f), 0)) for f in imagesFiles]
    #imgs = [cv2.resize(x, size) for x in imgs]

    return imagesFiles

def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    #linux
    fourcc = cv2.cv.CV_FOURCC(*'MP4V')
    filename = filename.replace('.mp4', '.avi')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)
    
    #mac
    #fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
    #video = cv2.VideoWriter()
    #success = video.open(filename, fourcc, fps, frame_size, True)
    #return video

def video_detect(model=None, probability_threshold=0.99999):
    INPUT_DIR = os.path.join("input", "video")
    OUTPT_DIR = os.path.join("graded_images", "video", "v.mp4")
    
    #read_image from folder
    f_list = load_images_from_dir(os.path.join(INPUT_DIR), ext=".jpg")
    f_list.sort()
    #print f_list

    #detected = []
    video_out = mp4_video_writer(OUTPT_DIR, (620, 771), fps=30)
    for f in f_list:
        print "currently processing: {}".format(f)
        image = cv2.imread(os.path.join(INPUT_DIR, f), -1)
        #cv2.imshow('dst_rt5', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        if model is not None: image_out = detect(image, model, probability_threshold)
        if image_out.shape != (620,771):
            image_out = image_out[0:771, 0:620]
        
        #detected.append(image_out)
        video_out.write(image_out)

    video_out.release()

def image_detect(model, filename, model_num, test_num, probability_threshold=0.99999):
    INPUT_DIR = "input"
    OUTPT_DIR = "graded_images"
    image = cv2.imread(os.path.join(INPUT_DIR, "{}.jpg".format(filename)), -1)
    #print image
    image_out = detect(image, model, probability_threshold)

    cv2.imwrite(os.path.join(OUTPT_DIR, "{}_model{}_{}.png".format(filename, model_num, test_num)), image_out)

def image_dir_detect(model, indir, outdir, probability_threshold=0.99999): 
    f_list = load_images_from_dir(os.path.join(indir), ext=".jpg")
    f_list.sort()
    i=1
    for f in f_list:
        image = cv2.imread(os.path.join(indir, f), -1)
        image_out = detect(image, model, probability_threshold)
        cv2.imwrite(os.path.join(outdir, "{}.png".format(i)), image_out)
        i += 1

if __name__ == "__main__":

    
    filename = "1"
    test_num = 1

    model_num = 5
    #weights = "model5weights1_80kneg_noresize.h5" #3
    weights = "model5epochweights2_40kneg_noresize_aug20_minus_mean.h5" #1
    #weights = "model5epochweights1_40kneg_noresize_aug20_minus_mean.h5" #2
    #weights = "model5epochweights2_40kneg_noresize_aug20_minus_mean.h5" #3
    #weights = "model5weights3_40kneg_noresize_aug40_minus_mean.h5" #3
    #weights = "model6weights1_40kneg_noresize_aug20_minus_mean.h5"
    #weights = "model5weights1_80kneg_noresize_aug20_minus_mean.h5"
    #weights = "model6epochweights1_40kneg_noresize_aug20_minus_mean.h5"

    exec("model = net{}()".format(model_num))
    model.load_weights(weights)
    sgd = keras.optimizers.SGD(lr=1e-2)#, decay=dacay, momentum=momentum, nesterov=nesterov)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    #image = cv2.imread(os.path.join(INPUT_DIR, "{}.jpg".format(filename)), -1) #flag -1: read image as is
    #image = cv2.imread("input/{}.jpg".format(filename), -1)
    #image_detect(model, filename, model_num, test_num, probability_threshold=0.99999)
    #video_detect(model, 0.999999)

    #detect all image in a folder
    image_dir_detect(model, "input", "graded_images", 0.99999)
   







