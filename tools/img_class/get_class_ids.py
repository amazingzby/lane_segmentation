import cv2
import os
import numpy as np
savd_path = "trainval/"
dataset_path = "/data/ai_lane/"
filename = "/data/ai_lane/trainval_list.txt"

#trainval id
# id 1 image number:10328
# id 2 image number:1392
# id 3 image number:11460
# id 4 image number:1060
# id 5 image number:153
# id 6 image number:393
# id 7 image number:377
# id 8 image number:1764
# id 9 image number:1281
# id 10 image number:591
# id 11 image number:14
# id 12 image number:59
# id 13 image number:483
# id 14 image number:141
# id 15 image number:62
# id 16 image number:45
# id 17 image number:21
# id 18 image number:15
# id 19 image number:2296

#val id
# id 1 image number:2092
# id 2 image number:275
# id 3 image number:2306
# id 4 image number:215
# id 5 image number:32
# id 6 image number:78
# id 7 image number:60
# id 8 image number:348
# id 9 image number:229
# id 10 image number:117
# id 11 image number:3
# id 12 image number:9
# id 13 image number:108
# id 14 image number:30
# id 15 image number:12
# id 16 image number:7
# id 17 image number:4
# id 18 image number:5
# id 19 image number:457

#train
# id 1 image number:8234
# id 2 image number:1117
# id 3 image number:9151
# id 4 image number:845
# id 5 image number:120
# id 6 image number:315
# id 7 image number:317
# id 8 image number:1416
# id 9 image number:1052
# id 10 image number:473
# id 11 image number:11
# id 12 image number:50
# id 13 image number:375
# id 14 image number:110
# id 15 image number:50
# id 16 image number:38
# id 17 image number:17
# id 18 image number:10
# id 19 image number:1839

#train B
# id 1 image number:8246
# id 2 image number:1131
# id 3 image number:9132
# id 4 image number:849
# id 5 image number:126
# id 6 image number:310
# id 7 image number:297
# id 8 image number:1402
# id 9 image number:1046
# id 10 image number:483
# id 11 image number:12
# id 12 image number:50
# id 13 image number:394
# id 14 image number:111
# id 15 image number:49
# id 16 image number:34
# id 17 image number:17
# id 18 image number:11
# id 19 image number:1859

#train_C
# id 1 image number:8257
# id 2 image number:1108
# id 3 image number:9145
# id 4 image number:844
# id 5 image number:124
# id 6 image number:315
# id 7 image number:310
# id 8 image number:1413
# id 9 image number:1007
# id 10 image number:467
# id 11 image number:8
# id 12 image number:40
# id 13 image number:379
# id 14 image number:112
# id 15 image number:51
# id 16 image number:37
# id 17 image number:19
# id 18 image number:14
# id 19 image number:1837

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
        print("id %d image number:%d"%(i,len(data)))
        np.save(npy_path,data)