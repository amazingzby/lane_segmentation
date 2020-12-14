import cv2
from glob import glob
#查看剪掉图片是否有正样本
if __name__ == "__main__":
    tag_files = glob("/data/ai_lane/trainval_tag/*.png")
    f = open("signal_croped.txt","w")
    for img_file in tag_files:
        img = cv2.imread(img_file,-1)
        h, w = img.shape[0], img.shape[1]
        if h == 720 and w == 1280:
            img_a = img[:208]
            # tag_a = tag[208:]
        elif h == 1080 and w == 1920:
            img_a = img[:312]
            # tag_a = tag[312:]
        else:
            print("%d shape error!" % filename)
            exit(0)
        labels_id = set(img_a.reshape(-1))
        if len(labels_id) >= 2:
            path = img_file.split("lane/")[-1]
            f.write(path+"\n")

