from __future__ import print_function

from scipy.ndimage.filters import gaussian_filter
from collections import defaultdict
from numpy import linalg as LA

import os
import numpy as np
import pandas as pd
import json
import skimage.measure, skimage.transform
import sys
import math

BODY_BONE = [1,2,3,4,5,6,7,8,9,10,11,12,13]
HEAD_BONE = [0,14,15,16,17]
MISSING_VALUE = -1

flames = 32
annotations = 18
coordinates = 2

def temp(pose_cords, flag = True):

    if flag == True:
        pose_cords = pose_cords[:,:,[1,0]]
        pose_cords[:,:,1] = 256 - pose_cords[:,:,1]

    else:
        pose_cords[:,:,1] = 256 - pose_cords[:,:,1]
        pose_cords = pose_cords[:,:,[1,0]]

    return pose_cords

def arccos(u, v):

    i = np.inner(u, v)
    n = LA.norm(u) * LA.norm(v)

    c = i / n
    a = np.arccos(np.clip(c, -1.0, 1.0))

    return a

def rotation(vec, t, deg=False):

    if deg == True:
        t = np.deg2rad(t)

    a = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    #print(t.shape)
    #print(vec.shape)

    ax = np.dot(a, vec)

    return ax

def coordinate_estimator(eye, err, base, correct_rad, frag = '14'):
    eye_vec = np.array([eye[0], eye[1]])
    err_vec = np.array([err[0], err[1]])
    base_vec = np.array([base[0], base[1]])

    eye_central = np.subtract(eye_vec, base_vec)
    err_central = np.subtract(err_vec, base_vec)

    if frag == '14':
        eye_rotate = rotation(eye_central, correct_rad)
        eye_correct = eye_rotate + base_vec

        return eye_correct

    elif frag == '15':
        eye_rotate = rotation(eye_central, correct_rad * -1)
        eye_correct = eye_rotate + base_vec

        return eye_correct

    elif frag == '16':
        eye_err_rad = arccos(err_central, eye_central)
        err_rotate = rotation(err_central, correct_rad + (2 * eye_err_rad))
        err_correct = err_rotate + base_vec

        return err_correct

    else:
        eye_err_rad = arccos(eye_central, err_central)
        err_rotate = rotation(err_central, -1 * (correct_rad + (2 * eye_err_rad)))
        err_correct = err_rotate + base_vec

        return err_correct


def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def flame_average(pose_cords):

    for i in range(flames):
        for j in range(annotations):

                if pose_cords[i,j,0] == MISSING_VALUE:
                    if i == 0:
                        if pose_cords[i+1,j,0] != MISSING_VALUE and pose_cords[i+2,j,0] != MISSING_VALUE:
                            pose_cords[i,j,:] = (pose_cords[i+1,j,:] + pose_cords[i+2,j,:])/2
                    elif not i == flames - 1:
                        if pose_cords[i-1,j,0] != MISSING_VALUE and pose_cords[i+1,j,0] != MISSING_VALUE:
                            pose_cords[i,j,:] = (pose_cords[i-1,j,:] + pose_cords[i+1,j,:])/2
                    else:
                        if pose_cords[i-1,j,0] != MISSING_VALUE and pose_cords[i-2,j,0] != MISSING_VALUE:
                            pose_cords[i,j,:] = (pose_cords[i-1,j,:] + pose_cords[i-2,j,:])/2


    return pose_cords

def head_anno_processing(pose_cords):

    count = 0
    correct_right = np.zeros(2)
    correct_left = np.zeros(2)
    correct_center = np.zeros(2)

    for i in range(flames):

        if pose_cords[i,14,0] != -1 and pose_cords[i,15,0] != -1:
            correct_right = correct_right + pose_cords[i,15,:]
            correct_left = correct_left + pose_cords[i,14,:]
            correct_center = correct_center + pose_cords[i,0,:]

            count += 1

    correct_right = correct_right / count
    correct_left = correct_left / count

    correct_center = correct_center / count

    #correct_rad = math.acos(np.linalg.norm(correct_left - correct_center, ord=2) /
                                #np.linalg.norm(correct_right - correct_center, ord=2))

    correct_rad = arccos(correct_left - correct_center, correct_right - correct_center)

    #print(np.linalg.norm(correct_left - correct_center, ord=2))

    for i in HEAD_BONE:
        for j in range(flames):

            if pose_cords[j,i,0] == -1 and i == 14:
                pose_cords[j,i,:] = coordinate_estimator(pose_cords[j,15,:], pose_cords[j,17,:], pose_cords[j,0,:], correct_rad, frag = '14')

            if pose_cords[j,i,0] == -1 and i == 15:
                pose_cords[j,i,:] = coordinate_estimator(pose_cords[j,14,:], pose_cords[j,16,:], pose_cords[j,0,:], correct_rad, frag = '15')

            if pose_cords[j,i,0] == -1 and i == 16:
                pose_cords[j,i,:] = coordinate_estimator(pose_cords[j,15,:], pose_cords[j,17,:], pose_cords[j,0,:], correct_rad, frag = '16')

            if pose_cords[j,i,0] == -1 and i == 17:
                pose_cords[j,i,:] = coordinate_estimator(pose_cords[j,14,:], pose_cords[j,16,:], pose_cords[j,0,:], correct_rad, frag = '17')

    return pose_cords

def pose_average(pose_cords):

    for i in range(flames):
        for j in range(annotations):

            if pose_cords[i,j,0] == MISSING_VALUE:

                sum_x, sum_y = 0, 0
                count = 0

                for k in range(annotations):

                    if pose_cords[i,k,0] != MISSING_VALUE:

                        sum_x += pose_cords[i,k,0]
                        sum_y += pose_cords[i,k,1]
                        count += 1

                x_value = sum_x / count
                y_value = sum_y / count

                pose_cords[i,j,:] = [x_value, y_value]

    return pose_cords

if __name__ == "__main__":


    arg = sys.argv
    input_annotation_folder = './annotations/' + arg[1] + '/'
    output_folder = './processed_anno/' + arg[1] + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for name in os.listdir(input_annotation_folder):

        df = pd.read_csv(input_annotation_folder + name, sep=':')
        df = df.sort_values('name')

        pose_cords = np.zeros((flames, annotations, coordinates))

        i = 0
        for index, row in df.iterrows():
            pose_cords[i] = load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])
            i += 1

        pose_cords = temp(pose_cords, flag = True)
        pose_cords = flame_average(pose_cords)
        pose_cords = head_anno_processing(pose_cords)
        pose_cords = pose_average(pose_cords)
        pose_cords = temp(pose_cords, flag = False)

        pose_cords = pose_cords.astype('int32')

        path = output_folder + name
        o_f = open(path, 'w')
        processed_names = set()
        print('name:keypoints_y:keypoints_x',file=o_f)
        for t in range(32):
            print('%s.jpg: %s: %s' % ('{0:02d}'.format(t), str(list(pose_cords[t, :, 0])), str(list(pose_cords[t, :, 1]))), file=o_f)
            o_f.flush()
