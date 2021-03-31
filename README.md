# Keras-FCN

This is a Keras implementation of the fully convolutional network outlined in Shelhamer et al. (2016), which performs semantic image segmentation according to the Pascal VOC Challenge.
This readme will present the results and then outline the basics of the implementation.
My hope is that this will be readable to non-technical persons, such as myself, who are looking to learn about fully convolutional networks.

## Introduction
The goal of **semantic segmentation** is to identify objects, such as cars or dogs, in an image and label the corresponding pixels.
For a quick introduction, see <a href="https://nanonets.com/blog/semantic-image-segmentation-2020/">this article</a>.

A **fully convolutional network (FCN)** does this by repurposing a convolutional neural network pre-trained to perform classification.
The CNN input is dragged across the image and the network tries to detect objects by classifying subregions of the image.
Doing this results in a 'heatmap' for regions where the CNN was able to arrive at a confident classification of an object.
However, this heatmap has low resolution since we downsample at each pooling layer of the CNN.
When we upsample to the original resolution, we can utilize the other layers of the CNN to increase the resolution of the heatmap.
For a quick introduction, see <a href="https://nanonets.com/blog/how-to-do-semantic-segmentation-using-deep-learning/">this article</a>.

As an example, below is an image and its label.

<img src="assets/biker.jpg" alt="biker" width=300> <img src="assets/biker_label.png" alt="biker label" width=300>

Below is the predicted label for naive FCNs trained to upsample by 32x, 16x, and 8x resolution.

<img src="assets/32.png" alt="32" width=300>

The <a href="http://host.robots.ox.ac.uk/pascal/VOC/">Pascal VOC project</a> is a dataset containing images that have been labeled.
There are 20 categories for the labels, which include aeroplanes, cars, people, and TVs.
The number of images with labels is augmented in the <a href="http://home.bharathh.info/pubs/codes/SBD/download.html">Berkeley Segmentation Boundaries Dataset</a>, which includes ~11k labelled images.

## Model

We follow the steps in the <a href="https://arxiv.org/abs/1605.06211">original paper by Shelhamer et al. (2016)</a>.
