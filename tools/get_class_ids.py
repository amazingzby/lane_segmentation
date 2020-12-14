import cv2
import os
import numpy as np
savd_path = "img_class/train/"
dataset_path = "/data/ai_lane/"
filename = "/data/ai_lane/train_list.txt"

if __name__ == "__main__":
    class_dict = {}
    for i in range(1,20):
        class_dict[i] = []
    print(class_dict)
    f = open(filename)
    lines = f.readlines()
    count = 1
    for line in lines:
        img_id = line.split()[1]
        img_id = img_id.replace("_crop","")
        savd_id = img_id.split("/")[-1][:-4]
        img = cv2.imread(os.path.join(dataset_path,img_id),-1)
        cls_ids = set(img.reshape(-1))
        for id in cls_ids:
            if id != 0:
                class_dict[id].append(savd_id)
        print("Processed %d images"%count)
        count += 1
    for i in range(1,20):
        data = np.array(class_dict[i])
        npy_path = os.path.join(savd_path,str(i)+".npy")
        np.save(npy_path,data)