from scipy.ndimage.filters import gaussian_filter
from collections import defaultdict

import os
import numpy as np
import pandas as pd
import json
import skimage.measure, skimage.transform
import sys

BODY_BONE = [1,2,3,4,5,6,7,8,9,10,11,12,13]
MISSING_VALUE = -1

def load_pose_cords(self, y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    cords = np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)
    return cords.astype(np.int)

def flame_average(pose_cords):

    for i in range(self.flames)
        for j in range(self.annotations):
            for k in range(self.coordinates):

                    if pose_cords[i,j,k] == MISSING_VALUE:
                        if i == 0:
                            if pose_cords[i+1,j,k] != MISSING_VALUE && pose_cords[i+2,j,k] != MISSING_VALUE:
                                pose_cords[i,j,k] = (pose_cords[i+1,j,k] + pose_cords[i+2,j,k])/2
                        elif not i == self.flames - 1:
                            if pose_cords[i-1,j,k] != MISSING_VALUE && pose_cords[i+1,j,k] != MISSING_VALUE:
                                pose_cords[i,j,k] = (pose_cords[i-1,j,k] + pose_cords[i+1,j,k])/2
                        else:
                            if pose_cords[i-1,j,k] != MISSING_VALUE && pose_cords[i-2,j,k] != MISSING_VALUE:
                                pose_cords[i,j,k] = (pose_cords[i-1,j,k] + pose_cords[i-2,j,k])/2


    return pose_cords

def head_anno_processing(pose_cords):

    return pose_cords

def pose_average(pose_cords):

    return pose_cords

def pose_mode(pose_cords):

    return pose_cords


if __name__ == "__main__":


    arg = sys.argv
    input_annotation_folder = './output/' + arg[1] + '/'
    output_folder = './processed_anno/' + arg[1] + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for name in os.listdir(input_annotation_folder)

        df = pd.read_csv(input_annotation_folder + name)

        for index, row in df.iterrows():
            pose_cords = load_pose_cords(row['keypoints_y'], row['keypoints_x'])

        pose_cords = flame_average(pose_cords)
        pose_cords =
