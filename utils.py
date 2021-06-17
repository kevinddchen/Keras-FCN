"""
This file contains useful methods for handling image files.
"""

import numpy as np
import tensorflow as tf
import cv2 as cv
import scipy.io # to read .mat files
from PIL import Image # to read raw data from palettised .png files

PALETTE = np.reshape([
    0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128,
    128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0,
    128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0,
    192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
    64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128,
    64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128,
    64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0,
    64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192,
    128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192,
    128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192,
    64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192,
    32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32,
    128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96,
    0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0,
    32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192,
    128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64,
    128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160,
    128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64,
    224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128,
    192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32,
    64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64,
    64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192,
    224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128,
    128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160,
    0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0,
    96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0,
    224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0,
    64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32,
    64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128,
    160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192,
    192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224,
    64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128, 224, 192, 64,
    96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192,
    64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160,
    0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224,
    32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128, 96, 160, 128,
    224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128,
    160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224,
    0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32,
    32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192,
    32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224,
    160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160, 192, 32, 96,
    64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32,
    224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224,
    64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224, 192], (-1, 3))



def get_image(path):
    '''Retrieve image as array of RGB values from .jpg file.
    
    Parameters
        path (string): Path to .jpg file
        
    Return
        (array<np.uint8>): RGB values for each pixel. Shape=(height, width, 3)
    '''
    arr = cv.imread(path)
    return np.flip(arr, axis=2) # OpenCV reads as BRG



def get_label_mat(path):
    '''Retrieve class labels for each pixel from Berkeley SBD .mat file.
    
    Parameters
        path (string): Path to .mat file
    
    Return
        (array<np.uint8>): Class as an integer in [0, 20] for each pixel. Shape=(height, width)
    '''
    mat = scipy.io.loadmat(path)
    return mat['GTcls']['Segmentation'].item(0,0) # this is how segmentation is stored



def get_label_png(path):
    '''Retrieve class labels for each pixel from Pascal VOC .png file.
    
    Parameters
        path (string): Path to .png file
    
    Return
        (array<np.uint8>): Class as an integer in [0, 20] or 255 (for boundary) for each pixel. Shape=(height, width)
    '''
    png = Image.open(path) # image is saved as palettised png. OpenCV cannot load without converting.
    return np.array(png)



def label_to_image(label, palette=PALETTE):
    '''Converts class labels to color image using a palette.
    
    Parameters
        label (array<np.uint8>): Class labels for each pixel. Shape=(height, width)
        palette (array<np.uint8>): RGB values for each class. Shape=(255, 3)
        
    Return
        (array<np.uint8>): RGB values for each pixel. Shape=(height, width, 3)
    '''
    return palette[label]



def label_to_onehot(label, num_classes=21):
    '''Converts class labels to its one-hot encoding.
    
    Parameters
        label (array<np.uint8>): Class labels for each pixel. Shape=(height, width)
        
    Return
        (array<np.uint8>): One-hot encoding of class labels for each pixel. Boundary (class 255) is ignored. 
                           Shape=(height, width, num_classes)
    '''
    return (np.arange(21) == label[..., None]).astype(np.uint8)



def onehot_to_label(arr):
    '''Opposite of label_to_onehot().'''
    return np.argmax(arr, axis=-1).astype(np.uint8)



## =======================
## Data augmentation
## =======================

def resize_and_pad(arr, x):
    '''Resize image into a square, keeping the original aspect ratio by center padding with black (or 
    boundary, if given class labels).
    
    Parameters
        arr (array<np.uint8>): RGB values or class labels for each pixel. Shape=(height, width[, 3])
        x (int): length of square
        
    Return
        (array<np.uint8>): Shape=(x, x[, 3])
    '''
    ## scale largest dimension to x
    h, w, *c = arr.shape
    f = min(float(x)/h, float(x)/w)
    arr = cv.resize(arr, None, fx=f, fy=f, interpolation=cv.INTER_NEAREST) # NEAREST important for class labels
                                                                           # images can use LINEAR instead
    ## pad with zeros
    h, w, *c = arr.shape
    if len(c) == 1: # RGB
        border = (0, 0, 0)
    else: # label
        border = 255
    h_pad, w_pad = x-h, x-w
    arr = cv.copyMakeBorder(arr, h_pad//2, (h_pad+1)//2, w_pad//2, (w_pad+1)//2, 
                            cv.BORDER_CONSTANT, value=border)
    return arr



## ==================================
## .tfrecords handling
## see tutorial: https://www.tensorflow.org/tutorials/load_data/tfrecord
## ==================================

def get_example(image, label):
    '''Given image and label, produce a tf Example that can be written to a .tfrecords file.
    
    Parameters
        image (array<np.uint8>): Shape=(height, width, 3)
        label (array<np.uint8>): Shape=(height, width)
        
    Return
        (tf Example)
    '''
    ## Usage:
    #with tf.io.TFRecordWriter(PATH_TO_TFRECORDS) as writer:
    #    writer.write(get_example(image, label).SerializeToString())
    feature = {
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))



def parse_example(example):
    '''Parse tf Example to obtain image and label.
    
    Parameters
        example (tf Example)
        
    Return
        image (array<np.uint8>): Shape=(height, width, 3)
        label (array<np.uint8>): Shape=(height, width)
    '''
    ## Usage:
    #dataset = tf.data.TFRecordDataset(PATH_TO_TFRECORDS).map(parse_example)
    feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    dct = tf.io.parse_single_example(example, feature_description)
    height = dct['height']
    width = dct['width']
    image = tf.reshape(tf.io.decode_raw(dct['image'], out_type=tf.uint8), (height, width, 3))
    label = tf.reshape(tf.io.decode_raw(dct['label'], out_type=tf.uint8), (height, width))
    return image, label
