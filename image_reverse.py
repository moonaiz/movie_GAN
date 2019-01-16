from PIL import Image, ImageOps
import os

input_folder = './R-hunds_up_images/'
sub_folder_list = os.listdir(input_folder)

for i in range(len(sub_folder_list)):
    sub_folder = input_folder + sub_folder_list[i] + '/'
    img_list = os.listdir(sub_folder)

    for j in range(len(img_list)):

        im = Image.open(sub_folder + img_list[j])
        im_mirror = ImageOps.mirror(im)
        im_mirror.save(sub_folder + img_list[j], quality=100)
