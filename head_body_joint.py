import sys
import os
import pandas as pd
import movie-head_gan

HEAD_BONE = [0,1,14,15,16,17]
BODY_BONE = [1,2,3,4,5,6,7,8,9,10,11,12,13]

flames = 32
annotations = 18
coordinates = 2

def joint(body, head):

if __name__ == '__main__':
    args = sys.argv
    input_body_folder = './output/' + args[1] + '_body/'
    input_head_folder = './output/' + args[1] + '_head/'

    head_annotations = np.zeros((flames, len(HEAD_BONE), coordinates))
    body_annotations = np.zeros((flames, len(BODY_BONE), coordinates))

    for name in os.listdir(input_body_folder):
            df_h = pd.read_csv(input_head_folder + '%s'% name, sep=':')
            df_h = df.sort_values('name')

            df_b = pd.read_csv(input_body_folder + '%s'% name, sep=':')
            df_b = df.sort_values('name')

            for index, row in df_h.iterrows():
                head.append(movie-head-gan.load_pose_cords(row['keypoints_y'], row['keypoints_x']))

            for index, row in df_b.itterrows():
                body.append(movie-head-gan.load_pose_cords(row['keypoints_y'], row['keypoints_x']))

            joint(body, head)

            
