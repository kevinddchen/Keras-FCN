"""
This file contains useful methods for handling image files.
"""

import numpy as np
import tensorflow as tf
import scipy.io # to read .mat files
from PIL import Image # to read image files

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
    jpg = Image.open(path).convert('RGB')
    return np.array(jpg)



def get_label_mat(path):
    '''Retrieve class labels for each pixel from Berkeley SBD .mat file.
    
    Parameters
        path (string): Path to .mat file
    
    Return
        (array<np.uint8>): Class as an integer in [0, 20] for each pixel. Shape=(height, width, 1)
    '''
    mat = scipy.io.loadmat(path)
    arr = mat['GTcls']['Segmentation'].item(0,0) # this is how segmentation is stored
    return arr[..., None]



def get_label_png(path):
    '''Retrieve class labels for each pixel from Pascal VOC .png file.
    
    Parameters
        path (string): Path to .png file
    
    Return
        (array<np.uint8>): Class as an integer in [-1, 20], where -1 is boundary, for each pixel. Shape=(height, width, 1)
    '''
    png = Image.open(path) # image is saved as palettised png. OpenCV cannot load without converting.
    arr = np.array(png)
    return arr[..., None]



def label_to_image(label, palette=PALETTE):
    '''Converts class labels to color image using a palette.
    
    Parameters
        label (array<np.uint8>): Class labels for each pixel. Shape=(height, width, 1)
        palette (array<np.uint8>): RGB values for each class. Shape=(255, 3)
        
    Return
        (array<np.uint8>): RGB values for each pixel. Shape=(height, width, 3)
    '''
    return palette[label[..., 0]]



def label_to_onehot(label, num_classes=21):
    '''Converts class labels to its one-hot encoding.
    
    Parameters
        label (array<np.uint8>): Class labels for each pixel. Shape=(height, width, 1)
        
    Return
        (array<np.uint8>): One-hot encoding of class labels for each pixel. Boundary is ignored. 
                           Shape=(height, width, num_classes)
    '''
    return np.arange(21) == label



def onehot_to_label(arr):
    '''Opposite of label_to_onehot().'''
    is_labelled = np.sum(arr, axis=-1)
    arr = np.argmax(arr, axis=-1).astype(np.uint8)
    arr[is_labelled == 0] = -1 # if pixel has no label, then it is boundary
    return arr[..., None]



## ==================================
## .tfrecords handling
## see tutorial: https://www.tensorflow.org/tutorials/load_data/tfrecord
## ==================================

def get_example(image, label):
    '''Given image and label, produce a tf Example that can be written to a .tfrecords file.
    
    Parameters
        image (array<np.uint8>): Shape=(height, width, 3)
        label (array<np.uint8>): Shape=(height, width, 1)
        
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
        label (array<np.uint8>): Shape=(height, width, 1)
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
    label = tf.reshape(tf.io.decode_raw(dct['label'], out_type=tf.uint8), (height, width, 1))
    return image, label
