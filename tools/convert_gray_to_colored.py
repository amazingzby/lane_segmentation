import numpy as np
import cv2
from PIL import Image

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

def gray2color(grayImg,num_classes):
    color_map = get_color_map_list(num_classes)
    lbl_pil = Image.fromarray(grayImg.astype(np.uint8), mode='P')
    lbl_pil.putpalette(color_map)
    #lbl_pil = cv2.cvtColor(np.array(lbl_pil),cv2.COLOR_RGB2BGR)
    return lbl_pil

if __name__ == "__main__":
    #imgPath = "/home/zby/data/dataset/ai_lane/train_tag/10013158.png"
    imgPath = "/home/zby/data/dataset/ai_lane/train_pic/10013158.jpg"
    img = cv2.imread(imgPath,-1)
    cv2.imshow("img",img)
    cv2.waitKey()
    #print(set(img.reshape(-1)))
    #gray2color(img,20)