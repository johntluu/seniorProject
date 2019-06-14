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

from api import PRN

from utils.rotate_vertices import frontalize
from utils.render_app import get_depth_image

def save_depth_image(depth_image):
    for line in depth_image:
        print(line)

def generate_depth_image(inputDir, prn):
    depth_images = []
    # os.environ['CUDA_VISIBLE_DEVICES'] = "-1" # only cpu for now
    # prn = PRN(is_dlib = True)

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
        if c>3:
            image = image[:,:,:3]

        # the core: regress position map
        max_size = max(image.shape[0], image.shape[1])
        if max_size> 1000:
            image = rescale(image, 1000./max_size)
            image = (image*255).astype(np.uint8)
        pos = prn.process(image) # use dlib to detect face
        
        image = image/255.
        if pos is None:
            continue
        vertices = prn.get_vertices(pos)
        save_vertices = frontalize(vertices)
        save_vertices[:,1] = h - 1 - save_vertices[:,1]

        depth = get_depth_image(vertices, prn.triangles, h, w)
        # save_depth_image(depth)
        depth_images.append(depth)
        return depth
        exit(1)
    print(len(depth_images))
    return depth_images[0]
    
def main():
    image = generate_depth_image()
