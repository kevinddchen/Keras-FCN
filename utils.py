"""
This file contains useful methods for training and running the FCN.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import tensorflow as tf

PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128,
    0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0,
    64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0,
    0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192,
    128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64,
    128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128,
    128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64,
    192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128,
    192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0,
    64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64,
    64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192,
    192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0,
    128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224,
    128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0,
    160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192,
    128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64,
    128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32,
    128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128,
    192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0,
    192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64,
    160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96,
    64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192,
    96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0,
    0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32,
    0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192,
    160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128,
    96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0,
    192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32,
    64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0,
    160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160,
    64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128,
    96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192,
    128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96,
    192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32,
    160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160,
    128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32,
    128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160,
    224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0,
    224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224,
    128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32,
    32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32,
    64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192,
    224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96,
    192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64,
    96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224,
    224, 192]



def get_image(path, resize='pad', out_dims=(512, 512)):
  """Retrieves image as array of RGB values.

    Parameters:

      path (string): Path to jpg/png image file
      resize (string): How image is resized 
        'pad': Resize by preserving aspect ratio, padding with zeros
        'resize': Resize by stretching to new dimensions
        None: Return original image without resizing
      out_dims (tuple): Dimensions of resized image

    Returns:

      NumPy array with shape=(*out_dims, 3)
  """

  img = Image.open(path).convert('RGB')

  if resize == 'resize': 
    img = img.resize(out_dims)
  elif resize == 'pad':
    img = ImageOps.pad(img, out_dims)
  elif resize == None:
    pass
  else:
    raise ValueError("option for 'resize' not recognized")

  return np.array(img)



def get_label_png(path, resize='pad', out_dims=(512, 512)):
  """Retrieves class labels for each pixel from Pascal VOC .png file.

    Parameters:

      path (string): Path to png file
      resize (string): How image is resized 
        'pad': Resize by preserving aspect ratio, padding with zeros
        'resize': Resize by stretching to new dimensions
        None: Return original image without resizing
      out_dims (tuple): Dimensions of resized image

    Returns:

      NumPy array with shape=out_dims
  """

  img = Image.open(path)

  if resize == 'resize': 
    img = img.resize(out_dims)
  elif resize == 'pad':
    img = ImageOps.pad(img, out_dims)
  elif resize == None:
    pass
  else:
    raise ValueError("option for 'resize' not recognized")

  arr = np.array(img)
  arr[arr==255] = 0 # convert border class (255) to background class (0)

  return arr



def get_label_mat(path, resize='pad', out_dims=(512, 512)):
  """Retrieves class labels for each pixel from Berkeley SBD .mat file.

    Parameters:

      path (string): Path to mat file
      resize (string): How image is resized
        'pad': Resize by preserving aspect ratio, padding with zeros
        'resize': Resize by stretching to new dimensions
        None: Return original image without resizing
      out_dims (tuple): Dimensions of resized image

    Returns:

      NumPy array with shape=out_dims
  """

  mat = scipy.io.loadmat(path)
  arr = mat['GTcls']['Segmentation'][0, 0]

  img = Image.fromarray(arr, mode='P')

  if resize == 'resize':
    img = img.resize(out_dims)
  elif resize == 'pad':
    img = ImageOps.pad(img, out_dims)
  elif resize == None:
    pass
  else:
    raise ValueError("option for 'resize' not recognized")

  arr = np.array(img)
  #no borders in Berkeley SBD
  #arr[arr==255] = 0 # convert border class (255) to background class (0)

  return arr



def display_image(arr, palette=PALETTE):
  """Display image from array.

    Parameters:

      arr (array): Array of integers specifying the image
        shape=(x, y): Entries are class labels
        shape=(x, y, 3): Entries are RGB values
        shape=(x, y, ?): Entries are one-hot encoding of class labels
      palette (list): 768 integerss specifying palette for image
  """
      
  arr = np.array(arr)

  if len(arr.shape) == 2:
    img = Image.fromarray(arr, mode='P')
    img.putpalette(palette)
    img = img.convert('RGB')
  elif arr.shape[-1] == 3:
    img = Image.fromarray(arr, mode='RGB')
  else:
    arr = de_one_hot_ize(arr)
    img = Image.fromarray(arr, mode='P')
    img.putpalette(palette)
    img = img.convert('RGB')

  plt.imshow(np.array(img))
  plt.axis('off')
  plt.show()



def one_hot_ize(arr, num_classes=21):
  """Convert array of class labels into one-hot encoding.

    Parameters:
      
      arr (array): Array of integer valued from 0 to (num_classes-1)
      num_classes (int): Number of classes

    Returns:

      NumPy array with shape=(*arr.shape, num_classes)
  """

  return (np.arange(num_classes) == arr[..., None]).astype(np.uint8)



def de_one_hot_ize(arr):
  """Opposite of one_hot_ize()."""

  return np.argmax(arr, axis=-1).astype(np.uint8)



## ============================================================================
## The following functions are adapted from the TensorFlow tutorial,
## https://www.tensorflow.org/tutorials/load_data/tfrecord

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

_feature_description = {
  'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
  'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
}

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  dct = tf.io.parse_single_example(example_proto, _feature_description)
  image = tf.reshape(tf.io.decode_raw(dct['image'], out_type=tf.uint8), (512, 512, 3))
  label = tf.reshape(tf.io.decode_raw(dct['label'], out_type=tf.uint8), (512, 512))
  return (image, label)

## ============================================================================



def write_to_record(record_file, file_names, image_dir, label_dir, 
                    image_ext='jpg', label_ext='mat', verbose=True, **kwargs):
  """Write images and labels to TFRecord file for fast access.

    Parameters:

      record_file (string): Path to save .tfrecords file
      file_names (string): Path to .txt file containing image/label file names
      image_dir (string): Directory containing image files
      label_dir (string): Directory containing label files
      image_ext (string): File extension of images
      label_ext (string): File extension of labels, either 'mat' or 'png'
      verbose (boolean): True to print status periodically
      Additional keyword arguments are passed to get_image and get_label calls
  """

  ## open .txt file and obtain all paths
  with open(file_names) as f:
    list_of_names = [s.rstrip('\n') for s in f.readlines()]

  N = len(list_of_names)

  ## write to TFRecord file
  with tf.io.TFRecordWriter(record_file) as writer:

    if verbose: print("Writing to file:", record_file)

    for i, s in enumerate(list_of_names):

      if verbose and i%100 == 0: print("... %d/%d" % (i, N))

      image = get_image(image_dir + '/' + s + '.' + image_ext, **kwargs)

      ## get image and label
      if label_ext == 'mat':
        label = get_label_mat(label_dir + '/' + s + '.' + label_ext, **kwargs)
      elif label_ext == 'png':
        label = get_label_png(label_dir + '/' + s + '.' + label_ext, **kwargs)
      else:
        raise ValueError("option for 'label_ext' not recognized")

      feature = {
          'image': _bytes_feature(image.tobytes()),
          'label': _bytes_feature(label.tobytes()),
      }
      tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
      writer.write(tf_example.SerializeToString())

    if verbose: print("Done!")



def read_from_record(record_file, parse_function=_parse_function):
  """Read images and labels from TFRecords file.

  Parameters:
  
    record_file (string): Path to .tfrecords file
    parse_function: Function to call on database

  Returns:

    TensorFlow Dataset that yields a tuple of numpy arrays (image, label).
    image has shape=(512, 512, 3) and label has shape=(512, 512), encoding the
    class label of each pixel as an integer.
  """

  return tf.data.TFRecordDataset(record_file).map(parse_function)

