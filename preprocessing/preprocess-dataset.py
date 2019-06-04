#!/usr/bin/env python
#######################################################################################
# The MIT License

# Copyright (c) 2014       Hannes Schulz, University of Bonn  <schulz@ais.uni-bonn.de>
# Copyright (c) 2013       Benedikt Waldvogel, University of Bonn <mail@bwaldvogel.de>
# Copyright (c) 2008-2009  Sebastian Nowozin                       <nowozin@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######################################################################################
# vim: set fileencoding=utf-8 :
#
# Helper script to convert the NYU Depth v2 dataset Matlab file into a set of
# PNG images in the CURFIL dataset format.
#
# See https://github.com/deeplearningais/curfil/wiki/Training-and-Prediction-with-the-NYU-Depth-v2-Dataset

from __future__ import print_function

from joblib import Parallel, delayed
from skimage import exposure
from skimage.io import imsave
import h5py
import numpy as np
import os
import png
import scipy.io
import sys
import cv2

from _structure_classes import get_structure_classes
import _solarized
from utils.rgbd_util import *
from utils.getCameraParam import *
from getHHA import *


### modified to collapse the ~950 original classes to 40 classes, not 4
### and to not put it in a color map format
def process_ground_truth(ground_truth):
    ### added:
    collapsed_classes = {
        # void is already 0
        "wall" : 1,
        "floor" : 2,
        "cabinet" : 3,
        "bed" : 4,
        "chair" : 5,
        "sofa" : 6,
        "table" : 7,
        "door" : 8,
        "window" : 9,
        "bookshelf" : 10,
        "picture" : 11,
        "counter" : 12,
        "blinds" : 13,
        "desk" : 14,
        "shelves" : 15,
        "curtain" : 16,
        "dresser" : 17,
        "pillow" : 18,
        "mirror" : 19,
        "floor mat" : 20,
        "clothes" : 21,
        "ceiling" : 22,
        "books" : 23,
        "refridgerator" : 24,
        "television" : 25,
        "paper" : 26,
        "towel" : 27,
        "shower curtain" : 28,
        "box" : 29,
        "whiteboard" : 30,
        "person" : 31,
        "night stand" : 32,
        "toilet" : 33,
        "sink" : 34,
        "lamp" : 35,
        "bathtub" : 36,
        "bag" : 37,
        #"otherstructure" : 38,
        #"otherfurniture" : 39,
        #"otherprop" : 40
    }
    ###

    ### anything commented out below is code I removed from the initial implementation
    #colors = dict()
    #colors["structure"] = _solarized.colors[5]
    #colors["prop"] = _solarized.colors[8]
    #colors["furniture"] = _solarized.colors[9]
    #colors["floor"] = _solarized.colors[1]
    shape = ground_truth.shape  # list(ground_truth.shape) + [3]
    img = np.ndarray(shape=shape, dtype=np.uint8)
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            l = ground_truth[i, j]
            #if (l == 0):
            #    img[i, j] = (0, 0, 0)  # background
            #else:
                #name = classes[names[l - 1]]
                #assert name in colors, name
                #img[i, j] = colors[name]

            name = names[l - 1]
            class_name = classes[name]
            if name in collapsed_classes:
                img[i, j] = collapsed_classes[name]
            elif class_name == "structure":
                img[i, j] = 38
            elif class_name == "furniture":
                img[i, j] = 39
            elif class_name == "prop":
                img[i, j] = 40
            else:
                img[i, j] = 0
    return img


def visualize_depth_image(data):

    data[data == 0.0] = np.nan

    maxdepth = np.nanmax(data)
    mindepth = np.nanmin(data)
    data = data.copy()
    data -= mindepth
    data /= (maxdepth - mindepth)

    gray = np.zeros(list(data.shape) + [3], dtype=data.dtype)
    data = (1.0 - data)
    gray[..., :3] = np.dstack((data, data, data))

    # use a greenish color to visualize missing depth
    gray[np.isnan(data), :] = (97, 160, 123)
    gray[np.isnan(data), :] /= 255

    gray = exposure.equalize_hist(gray)

    # set alpha channel
    gray = np.dstack((gray, np.ones(data.shape[:2])))
    gray[np.isnan(data), -1] = 0.5

    return gray * 255


### argument data_list_folder is added since the original. see main function
def convert_image(i, scene, img_depth, image, label,   data_list_folder):

    write_filenames = data_list_folder != None

    idx = int(i) + 1
    if idx in train_images:
        train_test = "training"
    else:
        assert idx in test_images, "index %d neither found in training set nor in test set" % idx
        train_test = "testing"

    folder = "%s/%s/%s" % (out_folder, train_test, scene)
    if not os.path.exists(folder):
        os.makedirs(folder)

    img_depth = img_depth * 1000.0

    depth_image_filename = "%s/%05d_depth.png" % (folder, i)
    png.from_array(img_depth, 'L;16').save(depth_image_filename)

    ### HHA processing added
    # depth image is in millimeters, and we need meters, so divide by 1000 ...
    D = img_depth / 1000.0  # lol

    HHA_depth_image_filename = "%s/%05d_depth_HHA.png" % (folder, i)
    HHA_depth_image = getHHA(getCameraParam('color'), D, D)
    cv2.imwrite(HHA_depth_image_filename, HHA_depth_image)

    ### block commented out
    #depth_visualization = visualize_depth_image(img_depth)
    #
    # workaround for a bug in the png module
    #depth_visualization = depth_visualization.copy()  # makes in contiguous
    #shape = depth_visualization.shape
    #depth_visualization.shape = (shape[0], np.prod(shape[1:]))
    #
    #depth_image = png.from_array(depth_visualization, "RGBA;8") 
    #depth_image.save("%s/%05d_depth_visualization.png" % (folder, i))

    image_filename = "%s/%05d_colors.png" % (folder, i)
    imsave(image_filename, image)

    ground_truth = process_ground_truth(label)
    ground_truth_filename = "%s/%05d_ground_truth.png" % (folder, i)
    imsave(ground_truth_filename, ground_truth)

    ### new:
    data_list_file = data_list_folder + train_test + ".txt"
    with open(data_list_file, 'a') as f:
        f.write("%s\t%s\t%s\n" % (image_filename, HHA_depth_image_filename, ground_truth_filename))
    ###


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("usage: %s <h5_file> <train_test_split> <out_folder> [<rawDepth> <num_threads>]" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    h5_file = h5py.File(sys.argv[1], "r")
    # h5py is not able to open that file. but scipy is
    train_test = scipy.io.loadmat(sys.argv[2])
    out_folder = sys.argv[3]

    ### added since original code:
    # folder to put training and testing data list files, training.txt and testing.txt
    # each contains a list of all filenames. each line: <rgb image>\t<depth image>\t<label image>\n
    data_list_folder = sys.argv[4]
    ###

    ### replaced since original code ...
    #if len(sys.argv) >= 5:
    #    raw_depth = bool(int(sys.argv[4]))
    #else:
    #    raw_depth = False
    raw_depth = False

    ### replaced since original code ...
    #if len(sys.argv) >= 6:
    #    num_threads = int(sys.argv[5])
    #else:
    #    num_threads = -1
    num_threads = 1

    test_images = set([int(x) for x in train_test["testNdxs"]])
    train_images = set([int(x) for x in train_test["trainNdxs"]])
    print("%d training images" % len(train_images))
    print("%d test images" % len(test_images))

    if raw_depth:
        print("using raw depth images")
        depth = h5_file['rawDepths']
    else:
        print("using filled depth images")
        depth = h5_file['depths']

    print("reading", sys.argv[1])

    labels = h5_file['labels']
    images = h5_file['images']

    rawDepthFilenames = [u''.join(unichr(c) for c in h5_file[obj_ref]) for obj_ref in h5_file['rawDepthFilenames'][0]]
    names = [u''.join(unichr(c) for c in h5_file[obj_ref]) for obj_ref in h5_file['names'][0]]
    scenes = [u''.join(unichr(c) for c in h5_file[obj_ref]) for obj_ref in h5_file['sceneTypes'][0]]
    rawRgbFilenames = [u''.join(unichr(c) for c in h5_file[obj_ref]) for obj_ref in h5_file['rawRgbFilenames'][0]]
    classes = get_structure_classes()

    print("processing images")
    if num_threads == 1:
        print("single-threaded mode")
        for i, image in enumerate(images):
            print("image", i + 1, "/", len(images))
            convert_image(i, scenes[i], depth[i, :, :].T, image.T, labels[i, :, :].T, data_list_folder)
    else:
        Parallel(num_threads)(delayed(convert_image)(i, scenes[i], depth[i, :, :].T, images[i, :, :].T, labels[i, :, :].T, data_list_folder) for i in range(len(images)))

    print("finished")
