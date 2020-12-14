import numpy as np
import os
import cv2
from PIL import Image
from datetime import datetime
import time
import sys

def getSecondElem(elem):
    return elem[1]

def morphology(grayImg):
    height = grayImg.shape[0]
    dstImg = np.zeros(grayImg.shape,grayImg.dtype)
    if height == 720:
        kernel = np.ones((7, 7), np.uint8)
    if height == 1080:
        kernel = np.ones((9,9),np.uint8)
    img_ids = set(grayImg.reshape(-1))
    img_ids.remove(0)
    id_nums = []
    for img_id in img_ids:
        cur_num = np.sum(grayImg == img_id)
        id_nums.append([img_id,cur_num])
    id_nums.sort(key=getSecondElem,reverse=True)
    for id_msg in id_nums:
        img_id = id_msg[0]
        cur_label = grayImg.copy()
        cur_label[cur_label != img_id] = 0
        cur_label = cv2.morphologyEx(cur_label, cv2.MORPH_CLOSE, kernel)
        dstImg[np.logical_and((cur_label!=0),(dstImg == 0))] = img_id
    return dstImg

def point_noise_remove(src_img):
    min_pixels1 = 100
    min_pixels2 = int(2.25*min_pixels1)
    height, width = src_img.shape
    pixels = set(src_img.reshape(-1))
    pixels.remove(0)
    for pixel in pixels:
        cur_label = src_img.copy()
        cur_label[cur_label != pixel] = 0
        #num_bbox联通区域数量，labels：不同标记
        num_bbox, labels, stats, centroids = cv2.connectedComponentsWithStats(cur_label, connectivity=8)
        for idx in range(num_bbox):
            num_pixels = np.sum(labels==idx)
            if (num_pixels < min_pixels1 and height == 720) or (num_pixels < min_pixels2 and height == 1080):
                src_img[labels==idx] = 0
    return src_img


if __name__ == "__main__":
    src_path = "result/result_img"
    dst_path = "result/result"
    count = 1
    filenames = os.listdir(src_path)
    for img_id in filenames:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),end=" ")
        print("Processing %d image:%s"%(count,img_id))

        grayImg = cv2.imread(os.path.join(src_path,img_id),-1)
        grayImg = morphology(grayImg)
        grayImg = point_noise_remove(grayImg)
        cv2.imwrite(os.path.join(dst_path,img_id),grayImg)
        count += 1
