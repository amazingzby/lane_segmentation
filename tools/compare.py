import cv2
import os
import numpy as np
from glob import glob
from tools.convert_gray_to_colored import gray2color

def get_color_map_list(num_classes):
    """ Returns the color map for visualizing the segmentation mask,
        which can support arbitrary number of classes.
    Args:
        num_classes: Number of classes
    Returns:
        The color map
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3

    return color_map

num_class = 20

val_files = glob("/data_bk/ai_lane/res_deeplab_medium/*.png")
labelpath = "/data/ai_lane/train_tag_crop/"

for sample in val_files:
    label_path = labelpath + sample.split("/")[-1]
    img1 = cv2.imread(sample,-1)
    img2 = cv2.imread(label_path,-1)
    img = cv2.hconcat([img1,img2])

    colored = gray2color(img, num_class)
    colored.save("temp/"+sample.split("/")[-1])



