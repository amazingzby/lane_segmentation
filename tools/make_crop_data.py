import os
import cv2

data_dir = "/data/ai_lane/"
train_pic_dir = os.path.join(data_dir,"trainval_pic")
train_tag_dir = os.path.join(data_dir,"trainval_tag")
test_pic_dir = os.path.join(data_dir,"testA")
train_pic_crop_dir = os.path.join(data_dir,"trainval_pic_crop")
train_tag_crop_dir = os.path.join(data_dir,"trainval_tag_crop")
test_pic_crop_dir = os.path.join(data_dir,"testA_crop")
#train
filenames = os.listdir(train_pic_dir)
for filename in filenames:
    img_id = filename[:-4]
    print("Process train image:%s"%filename)
    img = cv2.imread(os.path.join(train_pic_dir,img_id+".jpg"),-1)
    tag = cv2.imread(os.path.join(train_tag_dir,img_id+".png"),-1)
    h,w = img.shape[0],img.shape[1]
    if h==720 and w==1280:
        img_a = img[208:]
        tag_a = tag[208:]
    elif h==1080 and w == 1920:
        img_a = img[312:]
        tag_a = tag[312:]
    else:
        print("%d shape error!"%img_id)
        exit(0)
    cv2.imwrite(os.path.join(train_pic_crop_dir,img_id+".jpg"),img_a)
    cv2.imwrite(os.path.join(train_tag_crop_dir,img_id+".png"),tag_a)

#test
filenames = os.listdir(test_pic_dir)
for filename in filenames:
    print("Process test image:%s"%filename)
    img = cv2.imread(os.path.join(test_pic_dir,filename),-1)
    h,w = img.shape[0],img.shape[1]
    if h==720 and w==1280:
        img_a = img[208:]
        #tag_a = tag[208:]
    elif h==1080 and w == 1920:
        img_a = img[312:]
        #tag_a = tag[312:]
    else:
        print("%d shape error!"%filename)
        exit(0)
    cv2.imwrite(os.path.join(test_pic_crop_dir,filename),img_a)
