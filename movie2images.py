import numpy as np
import shutil
import cv2
import os

def movie2image(input_dir, movie_name, output_dir):
 part = 0
 num = 0
 video = cv2.VideoCapture(input_dir + movie_name)
 while(video.isOpened()):
    ret, frame = video.read()
    frame_sum = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)

    if ret == False:
        shutil.rmtree(output_dir + '/%s_part%s/'%(movie_name ,str(part)))
        break

    if num % 32 == 0:
        part += 1

    if not os.path.exists(output_dir + '/%s_part%s/'%(movie_name ,str(part))):
        os.makedirs(output_dir + '/%s_part%s/'%(movie_name ,str(part)))

    fix_frame = image_shape_fix(frame)

    cv2.imwrite(output_dir + '/%s_part%s/%s.jpg'% (movie_name, str(part), '{0:02d}'.format(num % 32)),fix_frame)
    num += 1

 video.release()

def image_shape_fix(input_img):
 img = input_img
 if img.shape[0] > img.shape[1]:
     long = img.shape[0]
     short = img.shape[1]
     fix_img = img[((long - short) // 2):((long - short)// 2 + short),:,:]
 else:
     long = img.shape[1]
     short = img.shape[0]
     fix_img = img[:, ((long - short) // 2):((long - short)// 2 + short),:]
 reshape_img = cv2.resize(fix_img, dsize = (256, 256))
 return reshape_img

if __name__ == "__main__":
 input_dir = './data/walk2/'
 output_dir = './images/walk2/'
 if not os.path.exists(output_dir):
    os.makedirs(output_dir)

 num = len(os.listdir(input_dir))

 for i in range (num):

    movie2image(input_dir, os.listdir(input_dir)[i], output_dir)
