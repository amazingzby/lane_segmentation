import cv2
import os
import numpy as np
from glob import glob
#      0.310 0.365 0.4153 0.3486 0.0000 0.0837 0.0000 0.4560 0.3342 0.1557 0.0000 0.0000 0.2951 0.0000 0.0000 0.0000 0.0000 0.0000 0.7499
#训练集[2524  693   2862   439    39     215    55     880    504    219    6      6      246    45     22     30     15     5      1149]
#验证集[226  69 235  41   1  20   4  67  48   7   1   0  24   6   3   3   2   0 86]
#new dataset
#[7737  642 8562  591  113  158  323  843  768  369    7   53  222   94 38   12    5   10 1075]
#total
#[10487, 1404, 11659, 1071, 153, 393, 382, 1790, 1320, 595, 14, 59, 492, 145, 63, 45, 22, 15, 2310]
#eval
#[1012    141   1137   110   19   27   42   177   136   55   3   6   50    8   7   5   1    1  219]
#repeat
#[285,    21,   429,    16,   0,   0,   7,   59,  45,   13,  0,  0,   8,   7,  1,  0,  1,  0,  15])
def show_one_image(img,label):
    img1 = img.copy().astype(np.float32)
    img2 = label.copy()
    height,width = img2.shape
    img1 /= 3
    for h in range(height):
        for w in range(width):
            if img2[h,w] != 0:
                img1[h,w,:] = [0,0,255]

    img1 = img1.astype(img.dtype)
    cv2.imshow("img",img1)
    cv2.waitKey()
if __name__ == "__main__":
    dataDir = "/data/ai_lane/"
    data_list = "/data/ai_lane/val_list.txt"

    cls_array = [0]*19
    count = 0
    # label_list = list(glob("/data/ai_lane/trainval_tag/*.png"))
    # for label_path in label_list:
    #     label = cv2.imread(label_path,-1)
    #     label = set(label.reshape(-1))
    #     for cls_id in label:
    #         if cls_id != 0:
    #             cls_array[cls_id-1] += 1
    #     count += 1
    #     if count % 100 == 0:
    #         print("%d images has processed!"%count)
    #         print(cls_array)
    # print(cls_array)
    #get cls num
    with open(data_list) as f:
        data = f.readlines()
        cls_array = np.zeros((19),dtype=np.int)
        for sample in data:
            img_path, label_path = os.path.join(dataDir, sample.split()[0]), \
                                   os.path.join(dataDir, sample.split()[1])
            label = cv2.imread(label_path,-1)
            label = set(label.reshape(-1))
            for cls_id in label:
                if cls_id != 0:
                    cls_array[cls_id-1] += 1
            print(cls_array)
