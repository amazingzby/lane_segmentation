import cv2
import os
import numpy as np

dir1 = "result/result"
dir2 = "result/result_submit"

if __name__ == "__main__":
    filenames = os.listdir(dir1)
    for img_id in filenames:
        img1 = cv2.imread(os.path.join(dir1,img_id),-1)
        img2 = cv2.imread(os.path.join(dir2,img_id),-1)
        if img1.shape[0] != img2.shape[0] or img1.shape[1] != img2.shape[1]:
            print(img1.shape)
            print(img2.shape)
            print("%s shape error!"%img_id)
            exit(0)
        diff = np.sum(img1 != img2)
        print(img_id + ",number of different pixels:  %d"%diff)
