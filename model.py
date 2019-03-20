'''
this script is used for setting up CNN model
training
validating
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as keras
#import tensorflow.contrib.keras.datasets
#from tensorflow.contrib.keras.models import Sequential
#from tensorflow.contrib.keras.layers.core import Flatten, Dense, Dropout
#from tensorflow.contrib.keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
#from tensorflow.contrib.keras.optimizers import SGD
#import cv2
import os
import sys
import argparse
#import matplotlib.pyplot as plt
import pandas as pd
import math
import h5py
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme, ShuffledExampleScheme
from fuel.datasets.hdf5 import H5PYDataset


def net1(weights_path=None): #full explicit vgg16
    model = keras.models.Sequential()
    model.add(keras.layers.Lambda(function=(lambda image: tf.image.resize_images(image, (48,48))), input_shape=(32,32,3)))
    model.add(keras.layers.ZeroPadding2D((1,1),input_shape=(48,48,3)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_normal', name='conv1'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv01'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv02'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv03'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv04'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv05'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv06'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv07'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv08'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv09'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv10'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv11'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv12'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu', name='fc1'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu', name='fc2'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(11, activation='softmax', name='predictions'))

    if weights_path:
        model.load_weights(weights_path)
    print model.summary()
    return model

def net2(load_weights=False, load_local_weights=True): #vgg16 modified
    if load_weights:
        if load_local_weights:
            model_vgg16_conv = keras.applications.vgg16.VGG16(weights=None, include_top=False) #weights='imagenet'
            model_vgg16_conv.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
        else:
             model_vgg16_conv = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False) #weights='imagenet'
    else:
        model_vgg16_conv = keras.applications.vgg16.VGG16(weights=None, include_top=False)
    #print model_vgg16_conv.summary()
    #freeze weights in vgg16
    #for layer in model_vgg16_conv.layers:
    #    layer.trainable = False
    
    input = keras.layers.Input(shape=(32,32,3),name='image_input')
    reshape_layer = keras.layers.Lambda(function=(lambda image: tf.image.resize_images(image, (48,48))), input_shape=(32,32,3))(input)
    output_vgg16_conv = model_vgg16_conv(reshape_layer)
    #Add the fully-connected layers 
    x = keras.layers.Flatten(name='flatten')(output_vgg16_conv)
    x = keras.layers.Dense(4096, activation='relu', name='fc1')(x)
    x = keras.layers.Dense(4096, activation='relu', name='fc2')(x)
    x = keras.layers.Dense(11, activation='softmax', name='predictions')(x)
    model = keras.models.Model(inputs=input, outputs=x)
    print model.summary()
    return model

def net3(weights_path=None): #custom net without non-digit class
    model = keras.models.Sequential()
    model.add(keras.layers.ZeroPadding2D((1,1),input_shape=(32,32,3)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv1'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv3'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv4'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu', name='fc1'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256, activation='relu', name='fc2'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation='softmax', name='predictions'))

    if weights_path:
        model.load_weights(weights_path)
    print model.summary()
    return model

def net4(weights_path=None): #custom net, include non-digit class in label
    model = keras.models.Sequential()
    model.add(keras.layers.ZeroPadding2D((1,1),input_shape=(32,32,3)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv1'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv3'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv4'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu', name='fc1'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(256, activation='relu', name='fc2'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(11, activation='softmax', name='predictions'))

    if weights_path:
        model.load_weights(weights_path)
    print model.summary()
    return model

def net5(weights_path=None): #custom net, include non-digit class in label
    model = keras.models.Sequential()
    model.add(keras.layers.ZeroPadding2D((1,1),input_shape=(32,32,3)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv1'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv3'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv4'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv5'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv6'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv7'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu', name='fc1'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(512, activation='relu', name='fc2'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(11, activation='softmax', name='predictions'))

    if weights_path:
        model.load_weights(weights_path)
    print model.summary()
    return model

def net6(weights_path=None): #full explicit vgg16 with 32x32 input
    model = keras.models.Sequential()
    #model.add(keras.layers.Lambda(function=(lambda image: tf.image.resize_images(image, (48,48))), input_shape=(32,32,3)))
    model.add(keras.layers.ZeroPadding2D((1,1),input_shape=(32,32,3)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_normal', name='conv1'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv01'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv02'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv03'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv04'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv05'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', name='conv06'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv07'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv08'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv09'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv10'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv11'))
    model.add(keras.layers.ZeroPadding2D((1,1)))
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', name='conv12'))
    model.add(keras.layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu', name='fc1'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu', name='fc2'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(11, activation='softmax', name='predictions'))

    if weights_path:
        model.load_weights(weights_path)
    print model.summary()
    return model

def dataset_generator(dataset, batch_size=500):
    while 1:
        trainstream = DataStream(dataset, iteration_scheme=ShuffledScheme(examples=dataset.num_examples, batch_size=batch_size))
        for data in trainstream.get_epoch_iterator():
            images, labels = data
            #resize images
            #images = tf.image.resize_images(images, (48,48))
            #images = images.eval()
            #standardize the input images
            m = images.mean(axis=(1,2,3), keepdims=True)
            #s = images.std(axis=(1,2,3), keepdims=True)
            images = (images - m)#/ (s + 1e-3)
            #change from "channel_first" to "channel_last"
            images = np.transpose(images, (0,2,3,1))
            #images = keras.utils.normalize(images, axis=-1, order=2)
            #convert to one_hot representation
            #print labels
            labels = keras.utils.to_categorical(labels, num_classes=11)
            #print labels
            yield(images, labels)
        trainstream.close()

def dataset_generator1(dataset, handle, batch_size=500):
    datagen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                                                            samplewise_center=False,
                                                            featurewise_std_normalization=False,
                                                            samplewise_std_normalization=True,
                                                            zca_whitening=False,
                                                            zca_epsilon=1e-6,
                                                            rotation_range=0.,
                                                            width_shift_range=0.,
                                                            height_shift_range=0.,
                                                            shear_range=0.,
                                                            zoom_range=0.,
                                                            channel_shift_range=0.,
                                                            fill_mode='nearest',
                                                            cval=0.,
                                                            horizontal_flip=False,
                                                            vertical_flip=False,
                                                            rescale=None,
                                                            preprocessing_function=None)

    handle = dataset.open()
    data = dataset.get_data(handle, request = range(dataset.num_examples))    

    stream = datagen.flow(data[0], data[1], batch_size=batch_size)
    dataset.close(handle)
    return stream

def step_decay(epoch): #step decay for learning rate
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def train(model=None):
    if model is not None:
        trainset = H5PYDataset('svhn_format_2.hdf5', which_sets=('train',), sources=('features', 'targets'))
        trainstream = DataStream(trainset, iteration_scheme=SequentialScheme(examples=trainset.num_examples, batch_size=500))
        for data in trainstream.get_epoch_iterator():
            images, labels = data
            #standardize the input images
            m = images.mean(axis=(2,3), keepdims=True)
            s = images.std(axis=(2,3), keepdims=True)
            images = (images - m)/s
            #change from "channel_first" to "channel_last"
            images = np.transpose(images, (0,2,3,1))
            labels = keras.utils.to_categorical(labels)
            #print images.shape
            model.train_on_batch(x=images, y=labels)
        trainstream.close()

def train1(model=None):
    if model is not None:
        trainset = H5PYDataset('svhn_format_2.hdf5', which_sets=('train',), sources=('features', 'targets'))
        testset = H5PYDataset('svhn_format_2.hdf5', which_sets=('test',), sources=('features', 'targets'))
        batch_size = 500
        epochs_to_wait_for_improve = 1
        csv_logger = keras.callbacks.CSVLogger('traininglog.csv')
        check_point = keras.callbacks.ModelCheckpoint("model3epochweights.h5", monitor='val_loss', 
                                                    verbose=0, save_best_only=False, 
                                                    save_weights_only=True, mode='auto', period=1)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs_to_wait_for_improve)
        history = model.fit_generator(dataset_generator(trainset, batch_size),
                                        steps_per_epoch=np.ceil(trainset.num_examples/batch_size), 
                                        epochs=15, verbose=2,
                                        callbacks=[csv_logger, check_point, early_stopping],
                                        validation_data=dataset_generator(testset, batch_size),
                                        validation_steps=np.ceil(testset.num_examples/batch_size))
        #print accuracy
        return history

def train2(model=None, num_epochs=1, epoch_weights="modelepochweights.h5", \
            weights="modelweights.h5", model_save="model.json",\
            log_save="modeltraininglog.csv"):
    if model is not None:
        dataset_size = 73257# + 531131 #this includes train (73257) and extra (531131)
        #use 20% as validation
        validation_size = int(0.2*dataset_size)
        train_size = dataset_size - validation_size
        #sequence of 1s and 0s for splitting dataset
        seq = np.hstack((np.zeros(validation_size),np.ones(train_size)))
        #randomize
        np.random.seed(1234)
        np.random.shuffle(seq)
        train_idx = np.where(seq==1)[0].tolist()
        validation_idx = np.where(seq==0)[0].tolist()

        trainset = H5PYDataset('svhn_format_2.hdf5', which_sets=('train',), 
                                sources=('features', 'targets'), subset=train_idx)
        validationset = H5PYDataset('svhn_format_2.hdf5', which_sets=('train',), 
                                sources=('features', 'targets'), subset=validation_idx)
        batch_size = 500
        epochs_to_wait_for_improve = 15
        csv_logger = keras.callbacks.CSVLogger(log_save)
        check_point = keras.callbacks.ModelCheckpoint(epoch_weights, monitor='val_loss', 
                                                        verbose=0, save_best_only=True, 
                                                        save_weights_only=True, mode='auto', period=1)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs_to_wait_for_improve)
        history = model.fit_generator(dataset_generator(trainset, batch_size),
                                        steps_per_epoch=np.ceil(trainset.num_examples/batch_size), 
                                        epochs=num_epochs, verbose=2,
                                        callbacks=[csv_logger, check_point, early_stopping],
                                        validation_data=dataset_generator(validationset, batch_size),
                                        validation_steps=np.ceil(validationset.num_examples/batch_size))
        save_model(model, weights, model_save)
        #print accuracy
        return history

def load_weights_to_model(model=None, weights_path=""):
    if weights_path and (model is not None):
        model.load_weights(weights_path)
        print "weights loaded from {}".format(weights_path)
    else:
        print "weights not loaded"

def train_no_aug(dataset_used, model=None, num_epochs=1, epoch_weights="modelepochweights.h5", \
            weights="modelweights.h5", model_save="model.json",\
            log_save="modeltraininglog.csv", \
            reduce_lr_patience=5, reduce_lr_min_lr=1e-7, reduce_lr_factor=0.5,
            early_stopping_patience=1, continue_training=False): #include neg samples
    if model is not None:
        #dataset_used = "new_more_neg.hdf5"
        dataset_size = H5PYDataset(dataset_used, which_sets=('train','train_neg')).num_examples#81257# + 531131 #this includes train (73257) and extra (531131), train_neg(8000), test_neg(2000)
        #use 20% as validation
        validation_size = int(0.2*dataset_size)
        train_size = dataset_size - validation_size
        #sequence of 1s and 0s for splitting dataset
        seq = np.hstack((np.zeros(validation_size),np.ones(train_size)))
        #randomize
        np.random.seed(1234)
        np.random.shuffle(seq)
        train_idx = np.where(seq==1)[0].tolist()
        validation_idx = np.where(seq==0)[0].tolist()

        trainset = H5PYDataset(dataset_used, which_sets=('train','train_neg'), 
                                sources=('features', 'targets'), subset=train_idx)
        validationset = H5PYDataset(dataset_used, which_sets=('train', 'train_neg'), 
                                sources=('features', 'targets'), subset=validation_idx)
        batch_size = 500
        #ksave training log
        csv_logger = keras.callbacks.CSVLogger(log_save, append=continue_training)
        #save weighst after each epoche
        check_point = keras.callbacks.ModelCheckpoint(epoch_weights, monitor='val_loss', 
                                                        verbose=0, save_best_only=False, 
                                                        save_weights_only=True, mode='auto', period=1)
        #reduce learning rate when validation loss is not decreasing
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor,
                              patience=reduce_lr_patience, min_lr=reduce_lr_min_lr)
        #step decreasing learning rate
        lrate = keras.callbacks.LearningRateScheduler(step_decay)
        #early stopping
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
        #tensor board
        tb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, \
                                        write_grads=False, write_images=False)
        callback_list = [reduce_lr, csv_logger, check_point, early_stopping, tb]

        history = model.fit_generator(dataset_generator(trainset, batch_size),
                                        steps_per_epoch=np.ceil(trainset.num_examples/batch_size), 
                                        epochs=num_epochs, verbose=1,
                                        callbacks=callback_list,
                                        validation_data=dataset_generator(validationset, batch_size),
                                        validation_steps=np.ceil(validationset.num_examples/batch_size))
        save_model(model, weights, model_save)
        #print accuracy
        return history



def train_aug(dataset_used, model=None, num_epochs=1, epoch_weights="modelepochweights.h5", \
            weights="modelweights.h5", model_save="model.json",\
            log_save="modeltraininglog.csv", \
            reduce_lr_patience=5, reduce_lr_min_lr=1e-7, reduce_lr_factor=0.5,
            early_stopping_patience=1, continue_training=False): #include neg samples, use augmentation by ImageDataGenerator
    if model is not None:
        traingen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                                                            samplewise_center=False,
                                                            featurewise_std_normalization=False,
                                                            samplewise_std_normalization=False,
                                                            zca_whitening=False,
                                                            zca_epsilon=1e-6,
                                                            rotation_range=20.,
                                                            width_shift_range=0.,
                                                            height_shift_range=0.,
                                                            shear_range=0.,
                                                            zoom_range=0.,
                                                            channel_shift_range=0.,
                                                            fill_mode='nearest',
                                                            cval=0.,
                                                            horizontal_flip=False,
                                                            vertical_flip=False,
                                                            rescale=None,
                                                            preprocessing_function=None)

        valgen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                                                            samplewise_center=False,
                                                            featurewise_std_normalization=False,
                                                            samplewise_std_normalization=False,
                                                            zca_whitening=False,
                                                            zca_epsilon=1e-6,
                                                            rotation_range=20.,
                                                            width_shift_range=0.,
                                                            height_shift_range=0.,
                                                            shear_range=0.,
                                                            zoom_range=0.,
                                                            channel_shift_range=0.,
                                                            fill_mode='nearest',
                                                            cval=0.,
                                                            horizontal_flip=False,
                                                            vertical_flip=False,
                                                            rescale=None,
                                                            preprocessing_function=None)

        #dataset_used = "new_more_neg.hdf5"
        
        dataset_size = H5PYDataset(dataset_used, which_sets=('train','train_neg')).num_examples#81257# + 531131 #this includes train (73257) and extra (531131), train_neg(8000), test_neg(2000)
        #use 20% as validation
        validation_size = int(0.2*dataset_size)
        train_size = dataset_size - validation_size
        #sequence of 1s and 0s for splitting dataset
        seq = np.hstack((np.zeros(validation_size),np.ones(train_size)))
        #randomize
        np.random.seed(1234)
        np.random.shuffle(seq)
        train_idx = np.where(seq==1)[0].tolist()
        validation_idx = np.where(seq==0)[0].tolist()

        trainsetX = H5PYDataset(dataset_used, which_sets=('train','train_neg'), 
                                sources=('features', ), subset=train_idx, load_in_memory=True)
        trainsetY = H5PYDataset(dataset_used, which_sets=('train','train_neg'), 
                                sources=('targets',), subset=train_idx, load_in_memory=True)
        validationsetX = H5PYDataset(dataset_used, which_sets=('train', 'train_neg'), 
                                sources=('features',), subset=validation_idx, load_in_memory=True)
        validationsetY = H5PYDataset(dataset_used, which_sets=('train', 'train_neg'), 
                                sources=('targets',), subset=validation_idx, load_in_memory=True)

        trainsetX, = trainsetX.data_sources
        validationsetX, = validationsetX.data_sources
        trainsetY, = trainsetY.data_sources
        validationsetY, = validationsetY.data_sources
        #sample wise center (all 3 channels)
        trainsetX = trainsetX - trainsetX.mean(axis=(1,2,3), keepdims=True)
        validationsetX = validationsetX - validationsetX.mean(axis=(1,2,3), keepdims=True)

        print trainsetX.shape
        print validationsetX.shape
        #images change to channel last
        trainsetX = np.transpose(trainsetX, (0,2,3,1))
        validationsetX = np.transpose(validationsetX, (0,2,3,1))

        #convert to one hot
        trainsetY = keras.utils.to_categorical(trainsetY, num_classes=11)
        validationsetY = keras.utils.to_categorical(validationsetY, num_classes=11)

        #traingen.fit(trainsetX)
        #valgen.fit(validationsetX)

        batch_size = 500
        #ksave training log
        csv_logger = keras.callbacks.CSVLogger(log_save, append=continue_training)
        #save weighst after each epoche
        check_point = keras.callbacks.ModelCheckpoint(epoch_weights, monitor='val_loss', 
                                                        verbose=0, save_best_only=False, 
                                                        save_weights_only=True, mode='auto', period=1)
        #reduce learning rate when validation loss is not decreasing
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor,
                              patience=reduce_lr_patience, min_lr=reduce_lr_min_lr)
        #step decreasing learning rate
        lrate = keras.callbacks.LearningRateScheduler(step_decay)
        #early stopping
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
        #tensor board
        tb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, \
                                        write_grads=False, write_images=False)
        callback_list = [reduce_lr, csv_logger, check_point, early_stopping, tb]

        history = model.fit_generator(traingen.flow(trainsetX, trainsetY, batch_size),
                                        steps_per_epoch=np.ceil(trainsetX.shape[0]/batch_size), 
                                        epochs=num_epochs, verbose=1,
                                        callbacks=callback_list,
                                        validation_data=valgen.flow(validationsetX, validationsetY, batch_size),
                                        validation_steps=np.ceil(validationsetX.shape[0]/batch_size))
        save_model(model, weights, model_save)
        #print accuracy
        return history

def test(model=None):
    if model is not None:
        #accuracies = []
        testset = H5PYDataset('svhn_format_2.hdf5', which_sets=('test',), sources=('features', 'targets'))
        teststream = DataStream(testset, iteration_scheme=SequentialScheme(examples=testset.num_examples, batch_size=500))
        for data in teststream.get_epoch_iterator():
            images, labels = data
            images = np.swapaxes(images, axis1=1, axis2=3)
            labels = keras.utils.to_categorical(labels)
            loss, accuracy = model.test_on_batch(x=images, y=labels)
            accuracies.append(accuracy)
        trainstream.close()
        return losses

def test1(model=None):
    if model is not None:
        #accuracies = []
        batch_size = 500
        testset = H5PYDataset('svhn_format_2.hdf5', which_sets=('test',), sources=('features', 'targets'))
        loss, accuracy = model.evaluate_generator(dataset_generator(testset, batch_size), 
                                                    steps=np.ceil(testset.num_examples/batch_size), 
                                                    max_queue_size=10, workers=1, 
                                                    use_multiprocessing=False)

        return loss, accuracy


def test_no_aug(dataset_used, model=None, testset=('test', 'test_neg',)): #include neg samples
    if model is not None:
        #accuracies = []
        #dataset_size = H5PYDataset('new.hdf5', which_sets=('test','test_neg')).num_examples
        #seq = np.arange(dataset_size)
        #np.random.seed(1234)
        #np.random.shuffle(seq)
        #test_idx=seq.tolist()
        batch_size = 500
        #dataset_used = "new_more_neg.hdf5"
        testset = H5PYDataset(dataset_used, which_sets=testset, sources=('features', 'targets')) 
        loss, accuracy = model.evaluate_generator(dataset_generator(testset, batch_size), 
                                                    steps=np.ceil(testset.num_examples/batch_size), 
                                                    max_queue_size=11, workers=1, 
                                                    use_multiprocessing=False)

        return loss, accuracy

def test_aug(dataset_used, model=None, testset=('test', 'test_neg',)): #include neg samples with augmentation
    if model is not None:
        #accuracies = []
        #dataset_size = H5PYDataset('new.hdf5', which_sets=('test','test_neg')).num_examples
        #seq = np.arange(dataset_size)
        #np.random.seed(1234)
        #np.random.shuffle(seq)
        #test_idx=seq.tolist()

        testgen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
                                                            samplewise_center=False,
                                                            featurewise_std_normalization=False,
                                                            samplewise_std_normalization=False,
                                                            zca_whitening=False,
                                                            zca_epsilon=1e-6,
                                                            rotation_range=20.,
                                                            width_shift_range=0.,
                                                            height_shift_range=0.,
                                                            shear_range=0.,
                                                            zoom_range=0.,
                                                            channel_shift_range=0.,
                                                            fill_mode='nearest',
                                                            cval=0.,
                                                            horizontal_flip=False,
                                                            vertical_flip=False,
                                                            rescale=None,
                                                            preprocessing_function=None)

        batch_size = 500
        #dataset_used = "new_more_neg.hdf5"
        
        testsetX = H5PYDataset(dataset_used, which_sets=testset, 
                                sources=('features', ), load_in_memory=True)
        testsetY = H5PYDataset(dataset_used, which_sets=testset, 
                                sources=('targets',), load_in_memory=True)
        testsetX, = testsetX.data_sources
        testsetY, = testsetY.data_sources
        #sample wise center (all 3 channels)
        testsetX = testsetX - testsetX.mean(axis=(1,2,3), keepdims=True)

        #images change to channel last
        testsetX = np.transpose(testsetX, (0,2,3,1))
        #one hot
        testsetY = keras.utils.to_categorical(testsetY, num_classes=11)
        #testgen.fit(testsetX)
        loss, accuracy = model.evaluate_generator(testgen.flow(testsetX, testsetY, batch_size), 
                                                    steps=np.ceil(testsetX.shape[0]/batch_size), 
                                                    max_queue_size=11, workers=1, 
                                                    use_multiprocessing=False)

        return loss, accuracy

def save_model(model, weights_path="", model_path=""):
    #save model structure to json
    if model_path:
        model_json = model.to_json()
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
    #save model weights to file
    if weights_path:
        model.save_weights(weights_path)

def train_model(dataset_used="new_more_neg.hdf5", model_num=1, attempt_num=None, comment="", weights_path="", num_epochs=100, \
                train_num=3, test_num=2, lr=1e-2, decay=1e-6, momentum=0.5, nesterov=True,\
                reduce_lr_patience=5, reduce_lr_min_lr=1e-7, reduce_lr_factor=0.5,\
                early_stopping_patience=50, continue_training=False):
    epoch_weights = "model{}epochweights{}_{}.h5".format(model_num, attempt_num, comment)
    weights = "model{}weights{}_{}.h5".format(model_num, attempt_num, comment)
    log_save = "model{}traininglog{}_{}.csv".format(model_num, attempt_num, comment)
    model_save = "model{}.json".format(model_num)

    tf.set_random_seed(0) # for reproducibility

    exec("model = net{}()".format(model_num))
    if weights_path:
        load_weights_to_model(model, weights_path)
        #for layer in model_vgg16_conv.layers:
        #    layer.trainable = False


    # Test pretrained model
    #model1 = net1('vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    #model1 = net1()
    #model1.load_weights('model1weights_save0.h5')
    #model2 = net2(load_weights=False, load_local_weights=False)
    #model3 = net3()
    #model4 = net4()
    #model.load_weights("model3weights.h5")
    #model4.load_weights("model4weights.h5")
    #model5 = net5()
    #model5.load_weights("model5weights_save3.h5")
    #sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #sgd = keras.optimizers.SGD(lr=1e-4, decay=1e-5, momentum=0.5, nesterov=True)
    sgd = keras.optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    #history1 = train3(model1, num_epochs=500, epoch_weights="model1epochweights.h5", \
    #        weights="model1weights.h5", model_save="model1.json", log_save="model1traininglog.csv")

    #history2 = train2(model2, num_epochs=100, epoch_weights="model2epochweights.h5", \
    #        weights="model2weights.h5", model_save="model2.json", log_save="model2traininglog.csv")

    #history3 = train2(model3, num_epochs=100, epoch_weights="model3epochweights.h5", \
    #        weights="model3weights.h5", model_save="model3.json", log_save="model3traininglog.csv")

    #history4 = train3(model4, num_epochs=100, epoch_weights="model4epochweights.h5", \
    #        weights="model4weights.h5", model_save="model4.json", log_save="model4traininglog.csv")
    print "learning rate is: {}".format(lr)
    exec("history = train{}(dataset_used=dataset_used, model=model, num_epochs=num_epochs, epoch_weights=epoch_weights, \
            weights=weights, model_save=model_save, log_save=log_save, \
            reduce_lr_patience=reduce_lr_patience, reduce_lr_min_lr=reduce_lr_min_lr, reduce_lr_factor=reduce_lr_factor,\
            early_stopping_patience=early_stopping_patience, continue_training=continue_training)".format(train_num))

    save_model(model, weights_path=weights, model_path=model_save)
    print "training completed, model saved to {}, weights saved to {}".format(model_save, weights)
    #testset = H5PYDataset('svhn_format_2.hdf5', which_sets=('test',), sources=('features', 'targets'))

    exec("loss, accuracy = test{}(dataset_used, model, testset=('test', 'test_neg',))".format(test_num))
    print "test loss:", loss
    print "test accuracy:", accuracy

    exec("loss, accuracy = test{}(dataset_used, model, testset=('test',))".format(test_num))
    print "pos test loss:", loss
    print "pos test accuracy:", accuracy

    exec("loss, accuracy = test{}(dataset_used, model, testset=('test_neg',))".format(test_num))
    print "neg test loss:", loss
    print "neg test accuracy:", accuracy

def test_model(dataset_used, model_file="", model_num=1, weights="", test_num=2):
    # load json and create model
    #json_file = open(model_file, 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #model = keras.models.model_from_json(loaded_model_json)
    exec("model = net{}()".format(model_num))

    #load weights into new model
    model.load_weights(weights)
    sgd = keras.optimizers.SGD(lr=1e-1)#, decay=dacay, momentum=momentum, nesterov=nesterov)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    exec("loss, accuracy = test{}(dataset_used, model, testset=('test', 'test_neg',))".format(test_num))
    print "test loss:", loss
    print "test accuracy:", accuracy

    exec("loss, accuracy = test{}(dataset_used, model, testset=('test',))".format(test_num))
    print "pos test loss:", loss
    print "pos test accuracy:", accuracy

    exec("loss, accuracy = test{}(dataset_used, model, testset=('test_neg',))".format(test_num))
    print "neg test loss:", loss
    print "neg test accuracy:", accuracy


if __name__ == "__main__":
    
    """
    comment: short comment to be append to the file names of weights, logs, etc
    model_num: which net to use
    attempt_num: a number to identify trials
    num_epochs: upper limit of total epochs
    train_num: which train method to use
    test_num: which test method to use
    lr: initial learning rate for optimizer
    decay: decay for optimizer
    momentum, nesterov: momentum parameter for sgd
    reduce_lr_patience: number of epochs to wait before reduce lr
    reduce_lr_min_lr: lower limit of lr
    refuce_lr_factor: reduce rate new_lr = lr * factor
    early_stopping_patience: number of epochs to wait before early stop
    continue_training: append csv log file if training is continuation of a previous one
    """
    task = "train"

    if task == "train":
        dataset_used = "new_more_neg_noresize.hdf5"
        comment = "40kneg_noresize_aug20_minus_mean"
        print "Note: {}".format(comment)
        train_model(dataset_used = dataset_used, model_num=5, attempt_num=1, comment=comment, \
                    weights_path="", num_epochs=300, \
                    train_num="_aug", test_num="_aug", lr=1e-2, decay=0, momentum=0.3, nesterov=True,\
                    reduce_lr_patience=5, reduce_lr_min_lr=1e-10, reduce_lr_factor=0.5,\
                    early_stopping_patience=20, continue_training=False)
        print "Note: {}".format(comment)
        print "data used: {}".format(dataset_used)
    

    if task == "test":
        #dataset_used = "new_more_neg.hdf5"
        #test_model(dataset_used, model_file="model1.json", model_num=2, weights="model2weights1_40kneg_pretrain_ft.h5", test_num="_no_aug") #2 total/pos/neg: 95/94/99
        
        #dataset_used = "new_more_neg_noresize.hdf5"
        #test_model(dataset_used, model_file="model1.json", model_num=5, weights="model5weights1_80kneg_noresize.h5", test_num="_no_aug") #1 total/pos/neg: 94/92/99
        #test_model(dataset_used, model_file="model1.json", model_num=5, weights="model5epochweights2_40kneg_noresize_aug20_minus_mean.h5", test_num="_aug") #total/pos/neg: 94/93/98
        #test_model(dataset_used, model_file="model1.json", model_num=5, weights="model5epochweights1_40kneg_noresize_aug20_minus_mean.h5", test_num="_aug") #total/pos/neg: 94/93/98
        #test_model(dataset_used, model_file="model1.json", model_num=5, weights="model5epochweights3_40kneg_noresize_aug20_minus_mean.h5", test_num="_aug") #total/pos/neg: 94/92/99
        #test_model(dataset_used, model_file="model1.json", model_num=5, weights="model5weights3_40kneg_noresize_aug40_minus_mean.h5", test_num="_aug") #total/pos/neg: 94/93/99
        #test_model(dataset_used, model_file="model1.json", model_num=5, weights="model6weights1_40kneg_noresize_aug20_minus_mean.h5", test_num="_aug") #total/pos/neg: 93/91/98

        dataset_used = "new_more_neg_noresize_80kneg.hdf5"
        #test_model(dataset_used, model_file="model1.json", model_num=5, weights="model5weights1_80kneg_noresize_aug20_minus_mean.h5", test_num="_aug") #total/pos/neg: 95/93/99
        test_model(dataset_used, model_file="model1.json", model_num=6, weights="model6weights1_80kneg_noresize_aug20_minus_mean.h5", test_num="_aug") #total/pos/neg: 94/91/99







