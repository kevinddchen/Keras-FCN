# WORK CURRENTLY IN PROGRESS!

# Keras-FCN

This is a Keras implementation of the fully convolutional network outlined in <a href="https://arxiv.org/abs/1605.06211">Shelhamer et al. (2016)</a>, which performs semantic image segmentation on the Pascal VOC dataset.
My hope is that this document will be readable to people outside of deep learning, such as myself, who are looking to learn about fully convolutional networks.

In preparation, I found the following repos invaluable for reference:

https://github.com/shelhamer/fcn.berkeleyvision.org

https://github.com/fmahoudeau/FCN-Segmentation-TensorFlow/

https://github.com/aurora95/Keras-FCN

## Introduction

The goal of **semantic segmentation** is to identify objects, like cars and dogs, in an image by labelling the corresponding groups of pixels according to their classes.
For an introduction, see <a href="https://nanonets.com/blog/semantic-image-segmentation-2020/">this article</a>.
As an example, below is an image and its labelled pixels.

| <img src="assets/rider.jpg" alt="biker" width=400> | <img src="assets/rider_label.png" alt="true label" width=400> |
|:---:|:---:|
| Image | True label |

A **fully convolutional network (FCN)** is an artificial neural network that performs semantic segmentation. 
The bottom layers of a FCN are those of a convolutional neural network (CNN), usually taken from a pre-trained network like VGGNet or GoogLeNet.
The purpose of these layers is to perform classification on subregions of the image.
The top layers of a FCN are **transposed convolution/deconvolution** layers, which upsample the results of the classification to the resolution of the original image.
This gives us a label for each pixel.
When upsampling, we can also utilize the intermediate layers of the CNN to improve the accuracy of the segmentation.
For an introduction, see <a href="https://nanonets.com/blog/how-to-do-semantic-segmentation-using-deep-learning/">this article</a>.

The <a href="http://host.robots.ox.ac.uk/pascal/VOC/">Pascal VOC project</a> is a dataset containing images whose pixels have been labeled according to 20 classes (plus the background), which include aeroplanes, cars, and people.
We will be performing semantic segmentation according to this dataset.

## Data

The number of images with labels in the Pascal VOC dataset is augmented by the <a href="http://home.bharathh.info/pubs/codes/SBD/download.html">Berkeley Segmentation Boundaries Dataset (SBD)</a>, which contains 11,355 labelled images.
However, there are 676 labelled images in the original Pascal VOC dataset that are missing from the SBD.
We have divided our data as follows:

- Training set: the SBD training set (8,498 images) + last 1,657 images (out of 2,857 total) of the SBD validation set + the 676 non-overlapping images of the Pascal VOC trainval set.
- Validation set: first 1,200 images (out of 2,857 total) of the SBD validation set

In total, we have 10,831 training images and 1,200 validation images.
The filenames of the training images are found in <a href="https://github.com/kevinddchen/Keras-FCN/blob/main/data/train_mat.txt">data/train_mat.txt</a> and <a href="https://github.com/kevinddchen/Keras-FCN/blob/main/data/train_png.txt">data/train_png.txt</a>. 
The filenames of the validation images are found in <a href="https://github.com/kevinddchen/Keras-FCN/blob/main/data/val_mat.txt">data/val_mat.txt</a>.
If you want to duplicate our dataset, you can download the <a href="https://github.com/kevinddchen/Keras-FCN/tree/main/data">data/</a> folder of this repository, which contains the 676 extra images of the Pascal VOC dataset, and the SBD dataset from their website.
After untarring, place the contents of `benchmark_RELEASE/dataset/img` into <a href="https://github.com/kevinddchen/Keras-FCN/tree/main/data/images_mat">data/images_mat/</a> and `benchmark_RELEASE/dataset/cls` into <a href="https://github.com/kevinddchen/Keras-FCN/tree/main/data/labels_mat">data/labels_mat/</a>.

<a href="https://github.com/kevinddchen/Keras-FCN/blob/main/data.ipynb">data.ipynb</a> puts the data into .tfrecords files, since it cannot all be loaded into RAM.

## Model

We followed the steps in the original paper.
Our model details can be found in <a href="https://github.com/kevinddchen/Keras-FCN/blob/main/models.py">models.py</a>.

The base CNN was VGG16.
First, the fully-connected layers were converted into convolutional layers.
Second, the final layer of VGG16 that predicted 1000 classes was replaced by a layer that predicted the 21 Pascal VOC classes (including the background).
Third, these predictions were fed into a deconvolution layer that upsampled 32x to the original resolution via bilinear interpolation.
This defines the **FCN32** network.

As previously mentioned, we utilized the intermediate layers of the CNN to improve the accuracy of the segmentation.
For the **FCN16** network, instead of upsampling 32x we first upsampled 2x to get an output whose resolution matched that of the `block4_pool` layer of VGG16.
We predicted 21 classes from `block4_pool` and added these two outputs together.
This was upsampled 16x to get to the original resolution.
A similar procedure was also done for the **FCN8** network, where we additionally included predictions from the `block3_pool` layer of VGG16.

The training details can be found in <a href="https://github.com/kevinddchen/Keras-FCN/blob/main/train.ipynb">train.ipynb</a>.
We trained each FCN32, FCN16, and FCN8 model from scratch for 20 epochs using the Adam optimizer at a fixed training rate of `1e-4`, with L<sup>2</sup> regularization with strength `1e-6`.
Some dropout is also added.

## Results

Below are the predicted labels for the example image above.

| <img src="assets/rider_label.png" alt="true label" width=200> | <img src="assets/fcn32.png" alt="fcn32 pred" width=200> |  |  |
| :--: | :--: | :--: | :--: |
| True label | FCN32 prediction | FCN16 prediction | FCN8 prediction |

The performance of these models on the validation set are summarized below.

| Model | Pixel Accuracy | Mean IoU |
| --- | --- | --- |
| FCN32 |  |  |
| FCN16 |  |  |
| FCN8 |  |  |

At the time of writing, the Pascal VOC website was down so I could not evaluate on the test set.

## Next Steps

I am quite happy with the performance of the models given the relatively simple implementation and short training period.
Our performance was comparable to that of Shelhamer, although we validated on a different dataset.
To get better performance, there are a couple of things that we can still do:

- Data set augmentation, such as cropping
- Use ensemble methods

When I have time, I will get to these.
