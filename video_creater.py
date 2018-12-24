import cv2
import os
import cmd import args

base_folder = './pose_check/'
input_folder = '%s/'% args.check_pose_dir

if not os.path.exists(base_folder + input_folder):
    print('--check_pose_dir ~~ is not exist')
    exit()

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video = cv2.VideoWriter('video.mp4', fourcc, 20.0, (128, 64))

for i in range(1, 20):
    img = cv2.imread('{0:02d}.jpg'.format(i))
    video.write(img)

video.release()
