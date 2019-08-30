import cv2
import os
from cmd import args

args = args()

base_folder = './pose_check/walk/'  + '%s/'% args.check_pose_dir + '/'
#input_folder = '%s/'% args.check_pose_dir

if not os.path.exists(base_folder):
    print('--check_pose_dir ~~ is not exist')
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('%s.mp4' % args.check_pose_dir, fourcc, 20.0, (256, 256))

for file_name in sorted(os.listdir(base_folder)):
    img = cv2.imread(base_folder + file_name)
    video.write(img)

video.release()
