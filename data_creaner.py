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

BODY_BONE = [1,2,3,4,5,6,7,8,9,10,11,12,13]

MISSING_VALUE = -1

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)


if __name__ == "__main__":
    import pandas as pd

    arg = sys.argv
    input_annotation_folder = './output/' + arg[1] + '/'
    output_folder = './processed_anno/' + arg[1] + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for name in os.listdir(input_annotation_folder)

        df = pd.read_csv(input_annotation_folder + name)

        for index, row in df.iterrows():
            pose_cords = load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])
