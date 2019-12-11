from scipy.ndimage.filters import gaussian_filter
from collections import defaultdict

import os
import numpy as np
import pandas as pd
import json
import skimage.measure, skimage.transform
import sys

BODY_BONE = [1,2,3,4,5,6,7,8,9,10,11,12,13]
HEAD_BONE = [0,14,15,16,17]
MISSING_VALUE = -1

flames = 32
annotations = 18
coordinates = 2

def load_pose_cords(self, y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    cords = np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)
    return cords.astype(np.int)

def flame_average(pose_cords):

    for i in range(flames)
        for j in range(annotations):
            for k in range(coordinates):

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

    for i in HEAD_BONE:
        for j in range(flames):

            if pose_cords[j,i,0] == -1 && i == 14:

                

            if pose_cords[j,i,0] == -1 && i == 15:

            if pose_cords[j,i,0] == -1 && i == 16:

                x_value = pose_codes[j,17,0] - pose_cords[j,0,0]
                y_value = pose_codes[j,17,1] - pose_cords[j,0,1]

                pose_cords[j,i,0] = pose_cords[j,0,0] - x_value
                pose_cords[j,i,1] = pose_cords[j,0,1] - y_value

            if pose_cords[j,i,0] == -1 && i == 17:

                x_value = pose_codes[j,16,0] - pose_cords[j,0,0]
                y_value = pose_codes[j,16,1] - pose_cords[j,0,1]

                pose_cords[j,i,0] = pose_cords[j,0,0] - x_value
                pose_cords[j,i,1] = pose_cords[j,0,1] - y_value

    return pose_cords

def pose_average(pose_cords):

    for i in range(flames):
        for j in range(annotations):

            if pose_cords[i,j,0] == MISSING_VALUE:

                sum_x, sum_y = 0
                count = 0

                for k in range(annotations):

                    if pose_cords[i,k,0] != MISSING_VALUE:

                        sum_x += pose_cords[i,k,0]sum_x += pose_cords[i,k,0]
                        sum_y += pose_cords[i,k,1]
                        count++

                x_value = sum_x / count
                y_value = sum_y / count

                pose_cords[i,j,:] = [x_value, y_value]

    return pose_cords


if __name__ == "__main__":


    arg = sys.argvsum_x += pose_cords[i,k,0]
    input_annotation_folder = './output/' + arg[1] + '/'
    output_folder = './processed_anno/' + arg[1] + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for name in os.listdir(input_annotation_folder):

        df = pd.read_csv(input_annotation_folder + name)

        for index, row in df.iterrows():
            pose_cords = load_pose_cords(row['keypoints_y'], row['keypoints_x'])

        pose_cords = flame_average(pose_cords)
        pose_cords = head_anno_processing(pose_cords)
        pose_cords = pose_average(pose_cords)

        o_f = open(output_folder + name, 'w')
        processed_names = set()
        print('name:keypoints_y:keypoints_x',file=o_f)
        for t in range(32):
            print('%s.jpg: %s: %s' % ('{0:02d}'.format(t), str(list(pose_cords[t, :, 0])), str(list(pose_cords[i, t, :, 1]))), file=o_f)
            result_file.flush()
