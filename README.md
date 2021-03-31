# Keras-FCN

This is a Keras implementation of the fully convolutional network outlined in Shelhamer et al. (2016), which performs semantic image segmentation according to the Pascal VOC Challenge.
This readme will present the results and then outline the basics of the implementation.
My hope is that this will be readable to non-technical persons, such as myself, who are looking to learn about fully convolutional networks.

## Introduction
The goal of **semantic segmentation** is to identify objects, such as cars or dogs, in an image and label the corresponding pixels.
For a quick introduction, see <a href="https://nanonets.com/blog/semantic-image-segmentation-2020/">this article</a>.

A **fully convolutional network (FCN)** does this by repurposing a convolutional neural network pre-trained to perform classification.
The CNN filters are dragged across the image and they try to detect objects by classifying subregions of the image.
Doing this results in a 'heatmap' around objects where the CNN was able to arrive at a confident classification.
This heatmap has low-resolution, since we downsampling at each pooling layer of the CNN.
When we upsample to the original resolution, we can utilize the other layers of the CNN to increase the resolution of the heatmap.
For a quick introduction, see <a href="https://nanonets.com/blog/how-to-do-semantic-segmentation-using-deep-learning/">this article</a>.

The <a href="http://host.robots.ox.ac.uk/pascal/VOC/">Pascal VOC project</a> has a dataset containing images that have been labeled.
There are 20 categories for the labels, which include aeroplanes, cars, people, and TVs.
The labelling of the original dataset was augmented in the <a href="http://home.bharathh.info/pubs/codes/SBD/download.html">Berkeley Segmentation Boundaries Dataset</a>, which includes ~11k labelled images.

As an example, below is an image and its label.

## Model

We follow the steps in the <a href="https://arxiv.org/abs/1605.06211">original paper by Shelhamer et al. (2016)</a>.
