#!/cm/shared/apps/virtualenv/csc/bin/python3.6
import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast

from PRNet.api import PRN

from PRNet.utils.rotate_vertices import frontalize
from PRNet.utils.render_app import get_depth_image

def initialize():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0" # 0 default gpu, -1 for cpu
    prn = PRN(is_dlib = True)
    return prn

def generate_depth_image(inputDir, prn):
    depth_images = []
    # not_detected_paths = []
    valid_paths = []

    types = ('*.jpg', '*.png')
    image_path_list= []

    for files in types:
        image_path_list.extend(glob(os.path.join(inputDir, files)))

    total_num = len(image_path_list)

    for i, image_path in enumerate(image_path_list):
        name = image_path.strip().split('/')[-1][:-4]

        # read image
        image = imread(image_path)
        [h, w, c] = image.shape

        if c > 3:
            image = image[:,:,:3]

        # the core: regress position map
        max_size = max(image.shape[0], image.shape[1])

        if max_size > 1000:
            image = rescale(image, 1000./max_size)
            image = (image*255).astype(np.uint8)

        pos = prn.process(image) # use dlib to detect face

        image = image/255.

        if pos is None:
            # not_detected_paths.append(image_path)
            continue
        
        vertices = prn.get_vertices(pos)
        save_vertices = frontalize(vertices)
        save_vertices[:,1] = h - 1 - save_vertices[:,1]

        depth = get_depth_image(vertices, prn.triangles, h, w)
        depth_images.append(depth)

        valid_paths.append(image_path)
    return depth_images, valid_paths
    # return depth_images, not_detected_paths

# unused
def generate_depth_images():
    rootDir = "./dataset/"
    depthImages = []
    allFilenames = []
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1" # only cpu for now
    prn = PRN(is_dlib = True)

    for root, dirs, files in os.walk(rootDir):
        for name in dirs:
            print(os.path.join(root, name))
            depthImage = generate_depth_image(os.path.join(root, name), prn)
            depthImages.append(depthImage)
    
    return depthImages

