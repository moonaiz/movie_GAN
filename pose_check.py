import pose_utils
import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.draw import circle, line_aa, polygon

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import skimage.measure, skimage.transform
import sys


LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
           [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
           [0,15], [15,17], [2,16], [5,17]]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
               'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

MISSING_VALUE = -1

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def draw_pose_from_cords(pose_joints, img_size, target_img, radius=2, draw_joints=True):
    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
    colors = target_img

    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
            to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
            colors[yy, xx] = np.expand_dims(val, 1) * 255

    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]

    return colors

if __name__ == "__main__":
    import pandas as pd
    import cv2
    from cmd import args

    args = args()
    input_image_folder = './images/walk2/'
    input_annotation_folder = './annotations/walk2/'
    output_folder = './pose_check/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sub_folder  = '%s/' % args.check_pose_dir

    if not os.path.exists(input_image_folder + sub_folder):
        print('images folder is nothing')
        exit()

    else:
        if not os.path.exists(output_folder + sub_folder):
            os.makedirs(output_folder + sub_folder)

    images_list = os.listdir(input_image_folder + sub_folder)

    df = pd.read_csv(input_annotation_folder + '%s.csv'% args.check_pose_dir, sep=':')

    for index, row in df.iterrows():
        pose_cords = load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])

        img = cv2.imread(input_image_folder + sub_folder + row['name'])
        colors= draw_pose_from_cords(pose_cords, (256, 256), img)

        cv2.imwrite(output_folder + sub_folder + row['name'], colors)
