import csv
import json
import numpy as np
import pandas as pd

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def reforme_index(pose_cords, y_max, x_max, y_min, x_min):
    y_cords = pose_cords[:,:,0]
    x_cords = pose_cords[:,:,1]

    if(y_max - y_min > x_max - x_min):
        base_len = y_max - y_min
    else:
        base_len = x_max - x_min

    #rescale 1 - 0
    y_cords = (y_cords - y_min) / base_len
    x_cords = (x_cords - x_min) / base_len

    y_cords = y_cords * (image_size[0] - 2 * space)
    y_cords += space
    x_cords = x_cords * (image_size[0] - 2 * space)
    x_cords += space

    return y_cords, x_cords

def min_max(pose_cords):
    y_max, x_max = 0, 0
    y_min, x_min = 255, 255
    for i in range(pose_cords.shape[0]):
        for j in range(pose_cords.shape[1]):
            if(y_max < pose_cords[i][j][0]):
                y_max = pose_cords[i][j][0]
            if(x_max < pose_cords[i][j][1]):
                x_max = pose_cords[i][j][1]
            if(y_min > pose_cords[i][j][0]):
                y_min = pose_cords[i][j][0]
            if(x_min > pose_cords[i][j][1]):
                x_min = pose_cords[i][j][1]


if __name__ == "__main__":
    from cmd import args

    args = args()
    input_annotation_folder = './handsup_annotations/'
    image_size = args.image_size

    df = pd.read_csv(input_annotation_folder + '%s.csv'% args.check_pose_dir, sep=':')

    pose_cords = np.zeros(shape=(32, 18, 2), dtype='float32')
    for i in range(32):
        for index, row in df.iterrows():
            pose_cord = load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])
        pose_cords[i] = pose_cord

    (y_max, x_max, y_min, x_min) = min_max(pose_cords)

    (y_cords, x_cords) = reforme_index(image_size, pose_cords, space, y_max, x_max, y_min, x_min)
    pose_cords[:,:,0] = y_cords
    pose_cords[:,:,1] = x_cords

    pose_cords = pose_cords.astype(np.uint32)

    output_path = input_annotation_folder + "%s-reforme.csv" %args.check_pose_dir

    result_file = open(output_path, 'w')
    processed_names = set()
    print("name:keypoints_y:keypoints_x",file=result_file)

    for t in range(32):
        print('%s.jpg: %s: %s' % ('{0:02d}'.format(t), str(list(pose_cords[t, :, 0])), str(list(pose_cords[t, :, 1]))), file=result_file)

    result_file.flush()
