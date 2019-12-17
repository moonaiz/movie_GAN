from scipy.ndimage.filters import gaussian_filter
from collections import defaultdict

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

def rotation(vec, t, deg=False):
    if deg == True:
        t = np.deg2rad(t)

    a = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    ax = np.dot(a, vec)

    return ax

def coordinate_estimator(eye, err, base, correct_rad, frag = 'eye'):
    eye_central = eye - base
    err_central = err - base

    eye_rad = math.acos(np.linalg.norm(np.array([0, eye_central[1]]), ord=2) /
                                            np.linalg.norm(eye_central, ord=2))

    err_rad = math.acos(np.linalg.norm(np.array([0, err_central[1]]), ord=2) /
                                            np.linalg.norm(err_central, ord=2))

    fix_rad = eye_rad - correct_rad

    eye_central = rotation(eye_central, fix_rad)
    err_central = rotation(err_central, fix_rad)

    eye_reverse = np.array([eye_central[0] * -1, eye_central[1]])
    err_reverse = np.array([err_central[0] * -1, err_central[1]])

    eye_reverse = rotation(eye_reverse, fix_rad * -1)
    err_reverse = rotation(err_reverse, fix_rad * -1)

    eye_reverse = eye_reverse + base
    err_reverse = err_reverse + base

    if frag == 'eye':
        return eye_reverse
    else:
        return err_reverse


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

    count = 0
    correct_right = np.zeros(2,)
    correct_left = np.zeros(2,)
    correct_center = np.zeros(2,)

    for i in range(flames):

        if pose_codes[i,14,0] != -1 && pose_codes[i,15,0] != -1:
            correct_right += pose_codes[i,15,:]
            correct_left += pose_codes[i,14,:]
            correct_center += pose_codes[i,0,:]

            count += 1

        correct_right = correct_right / count
        correct_left = correct_left / count
        correct_center = correct_center / count

        correct_rad = math.acos(np.linalg.norm(correct_left - correct_center, ord=2) /
                                            np.linalg.norm(correct_right - correct_center, ord=2))


    for i in HEAD_BONE:
        for j in range(flames):

            if pose_cords[j,i,0] == -1 && i == 14:
                pose_codes[j,i,:] = coordinate_estimator(pose_codes[j,15,:], pose_codes[j,17,:], pose_codes[j,0,:],
                                                            correct_rad, frag = 'eye')

            if pose_cords[j,i,0] == -1 && i == 15:
                pose_codes[j,i,:] = coordinate_estimator(pose_codes[j,14,:], pose_codes[j,16,:], pose_codes[j,0,:],
                                                            correct_rad, frag = 'eye')

            if pose_cords[j,i,0] == -1 && i == 16:
                pose_codes[j,i,:] = coordinate_estimator(pose_codes[j,15,:], pose_codes[j,17,:], pose_codes[j,0,:],
                                                            correct_rad, frag = 'err')

            if pose_cords[j,i,0] == -1 && i == 17:
                pose_codes[j,i,:] = coordinate_estimator(pose_codes[j,14,:], pose_codes[j,16,:], pose_codes[j,0,:],
                                                            correct_rad, frag = 'err')
                                                            
                """""
                x_value = pose_codes[j,17,0] - pose_cords[j,0,0]
                y_value = pose_codes[j,17,1] - pose_cords[j,0,1]

                pose_cords[j,i,0] = pose_cords[j,0,0] - x_value
                pose_cords[j,i,1] = pose_cords[j,0,1] - y_value
                x_value = pose_codes[j,16,0] - pose_cords[j,0,0]
                y_value = pose_codes[j,16,1] - pose_cords[j,0,1]

                pose_cords[j,i,0] = pose_cords[j,0,0] - x_value
                pose_cords[j,i,1] = pose_cords[j,0,1] - y_value
                """""

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

"""

import numpy as np
import matplotlib.pyplot as plt
import math
%matplotlib inline

eye_r = np.array([[5, 5],[4,7]])
err_r = np.array([[5, 5],[2,6]])

y_vec = np.array([[5, 5], eye_r[:,1]])

plt.plot(eye_r[:, 0], eye_r[:, 1], label='eye_r')
plt.plot(err_r[:, 0], err_r[:, 1], label='err_r')

eye_angle = math.acos(np.linalg.norm(y_vec[1] - y_vec[0], ord=2) / np.linalg.norm(eye_r[1] - eye_r[0], ord=2))
err_angle = math.acos(np.linalg.norm(y_vec[1] - y_vec[0], ord=2) / np.linalg.norm(err_r[1] - err_r[0], ord=2))

print(eye_angle)
print(err_angle)



plt.legend()

plt.show()

"""

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
