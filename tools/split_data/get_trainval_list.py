import os
import hashlib
import numpy as np

data_dir = "/data/ai_lane/"
img_dir = "trainval_pic/"
tag_dir = "trainval_tag/"

#总共15503张图片，经处理后有15265张，重复238个

def cal_md5(filename):
    data = open(data_dir+img_dir+filename, 'rb')
    buf1 = data.read()
    hasher = hashlib.md5()
    hasher.update(buf1)
    md5 = hasher.hexdigest()
    return md5

if __name__ == "__main__":
    path1 = os.path.join(data_dir,img_dir)
    path2 = os.path.join(data_dir,tag_dir)
    img_list = np.array(list(os.listdir(path1)))
    f_trainval = open(os.path.join(data_dir,"trainval_list.txt"),"w")
    num_total = len(img_list)
    print("Total images:%d"%num_total)
    md5_list = []
    count = 0
    for imgfile in img_list:
        img_md5 = cal_md5(imgfile)
        md5_list.append(img_md5)
        count += 1
        if count % 1000 == 0:
            print("%d md5 has been processed!"%count)
    count = 0
    for idx1 in range(num_total):
        img_name = img_list[idx1]
        same_md5 = False
        count += 1
        for idx2 in range(idx1+1,num_total):
            if md5_list[idx1] == md5_list[idx2]:
                same_md5 = True
                break
        if not same_md5:
            name1 = os.path.join(img_dir,img_name)
            name2 = os.path.join(tag_dir,img_name[:-4]+".png")
            f_trainval.write(name1 + " " + name2 + "\n")
        if count % 1000 == 0:
            print("%d images has been written!"%count)





    # count = 1
    # for img_name in img_list:
    #     print("Processing %d image:%s"%(count,img_name))
    #     img1 = cv2.imread(os.path.join(data_dir+"/"+img_dir,img_name),-1)
    #     img2 = cv2.imread(os.path.join(data_dir+"/"+tag_dir,img_name[:-4]+".png"),-1)
    #     if (img1.shape[0] == img2.shape[0]) and (img1.shape[0] == img2.shape[0]):
    #         name1 = os.path.join(img_dir,img_name)
    #         name2 = os.path.join(tag_dir,img_name[:-4]+".png")
    #         if count <= 14000:
    #             f_train.write(name1+" "+name2+"\n")
    #         else:
    #             f_eval.write(name1+" "+name2+"\n")
    #     else:
    #         name1 = os.path.join(img_dir,img_name)
    #         name2 = os.path.join(tag_dir,img_name[:-4]+".png")
    #         f_error.write(name1+" "+name2+"\n")
    #     count += 1
    # f_train.close()
    # f_eval.close()
    # f_error.close()
