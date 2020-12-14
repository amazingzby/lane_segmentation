import os
import cv2
import numpy as np

data_dir = "/data/ai_lane"
img_dir = "trainval_pic"
tag_dir = "trainval_tag"

if __name__ == "__main__":
    path1 = os.path.join(data_dir,img_dir)
    path2 = os.path.join(data_dir,tag_dir)
    f_trainval = open(os.path.join(data_dir,"trainval_list.txt"))
    lines = f_trainval.readlines()
    img_list = np.array(list(lines))
    for _ in range(10):
        np.random.shuffle(img_list)
    f_train = open(os.path.join(data_dir,"train_list.txt"),"w")
    f_eval = open(os.path.join(data_dir,"val_list.txt"),"w")
    num_total = len(img_list)
    num_train = int(num_total*0.8)
    count = 0
    for msg in img_list:
        print("Processing %d images"%(count))
        if count <= num_train:
            f_train.write(msg)
        else:
            f_eval.write(msg)
        count += 1
    f_train.close()
    f_eval.close()