import csv
import pandas as pd

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def reforme_index(pose_cords, y_max, x_max, y_min, x_min):
    for i in range(pose_cords.shape[0]):
        for j in range(pose_cords.shape[1]):
            y_cords = pose_cords[i][j][0]
            x_cords = pose_cords[i][j][1]

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
    input_annotation = './output/annotations/'

    df = pd.read_csv(input_annotation_folder + '%s.csv'% args.check_pose_dir, sep=':')

    pose_cords = np.zeros(shape=(32, 18, 2), dtype='float32')
    for i in range(32):
        for index, row in df.iterrows():
            pose_cord = load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])
        pose_cords[i] = pose_cord

    y_max, x_max, y_min, x_min = min_max(pose_cords)
