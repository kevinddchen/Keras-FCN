'''
This file contains methods for dataset augmentation.
'''

import numpy as np
import tensorflow as tf

def resize_with_pad(image, label, size=512):
    '''Resize a square while keeping the original aspect ratio, padding with black for the image and boundary 
    for the label.
    
    Args:
      image (array<np.uint8>): RGB values for each pixel. Shape=(height, width, 3)
      label (array<np.uint8>): Class labels for each pixel. Shape=(height, width, 1)
      size (int): length of square
        
    Returns:
      (array<np.uint8>): Resized image. Shape=(size, size, 3)
      (array<np.uint8>): Resized label. Shape=(size, size, 1)
    '''
    image = tf.image.resize_with_pad(image, size, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    ## since `resize_with_pad` pads with zeros, use fact that boundary class is -1 to pad with -1 instead.
    label = tf.image.resize_with_pad(label+1, size, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)-1
    return image, label
