import os
import cv2
import numpy as np

data_dir = "/data/ai_lane"
img_dir = "trainval_pic"
tag_dir = "trainval_tag"
test_dir= "testA_crop"

if __name__ == "__main__":
    path1 = os.path.join(data_dir,img_dir)
    path2 = os.path.join(data_dir,tag_dir)
    img_list = np.array(list(os.listdir(path1)))
    for _ in range(5):
        np.random.shuffle(img_list)
    print(len(img_list))
    f_train = open(os.path.join(data_dir,"train_list.txt"),"w")
    f_eval = open(os.path.join(data_dir,"val_list.txt"),"w")
    f_error = open(os.path.join(data_dir,"error_list.txt"),"w")
    count = 1
    for img_name in img_list:
        print("Processing %d image:%s"%(count,img_name))
        img1 = cv2.imread(os.path.join(data_dir+"/"+img_dir,img_name),-1)
        img2 = cv2.imread(os.path.join(data_dir+"/"+tag_dir,img_name[:-4]+".png"),-1)
        if (img1.shape[0] == img2.shape[0]) and (img1.shape[0] == img2.shape[0]):
            name1 = os.path.join(img_dir,img_name)
            name2 = os.path.join(tag_dir,img_name[:-4]+".png")
            if count <= 14000:
                f_train.write(name1+" "+name2+"\n")
            else:
                f_eval.write(name1+" "+name2+"\n")
        else:
            name1 = os.path.join(img_dir,img_name)
            name2 = os.path.join(tag_dir,img_name[:-4]+".png")
            f_error.write(name1+" "+name2+"\n")
        count += 1
    f_train.close()
    f_eval.close()
    f_error.close()
