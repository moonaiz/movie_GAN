import sys
import os
import pandas as pd

HEAD_BONE = [0,1,14,15,16,17]
BODY_BONE = [1,2,3,4,5,6,7,8,9,10,11,12,13]

def joint(body, head):

if __name__ == '__main__':
    args = sys.argv
    input_body_folder = './output/' + args[1] + '_body/'
    input_head_folder = './output/' + args[1] + '_head/'

    for name in os.listdir(input_body_folder):
            df = pd.read_csv(input_body_folder + '%s'% name, sep=':')
            df = df.sort_values('name')

            for index, row in df.iterrows():
                x_str = row['keypoints_x']
