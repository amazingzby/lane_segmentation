import cv2
import hashlib
trainval_dir = "/data/ai_lane/trainval_pic/"
#org
#[656, 406, 29, 573, 28, 0, 0, 10, 75, 80, 16, 0, 0, 17, 8, 2, 0, 2, 0, 26]
#去重
#[172, 121, 8, 144, 12, 0, 0, 3, 16, 35, 3, 0, 0, 9, 1, 1, 0, 1, 0, 11]
if __name__ == "__main__":
    f = open("same_train.txt")
    samples = f.readlines()
    f.close()
    class_ids = [0]*20
    # for sample in samples:
    #     id1,id2 = sample.split()[:2]
    #     img1 = cv2.imread(trainval_dir+id1[:-3]+"png",-1)
    #     img2 = cv2.imread(trainval_dir+id2[:-3]+"png",-1)
    #     id_set1 = set(img1.reshape(-1))
    #     id_set2 = set(img2.reshape(-1))
    #     for id in id_set1:
    #         class_ids[id]+=1
    #     for id in id_set2:
    #         class_ids[id] += 1
    # print(class_ids)
    img_list = []
    for sample in samples:
        img_id = sample.split()[0]
        img_list.append(img_id)
    for i in range(len(img_list)-1,0,-1):
        path1 = trainval_dir + img_list[i]
        data1 = open(path1,'rb')
        buf1 = data1.read()
        hasher1 = hashlib.md5()
        temp = hasher1.update(buf1)
        temp_md51 = hasher1.hexdigest()

        for j in range(i-1,-1,-1):
            path2 = trainval_dir+img_list[j]
            if path1 == path2:
                img_list.pop(i)
                break
            data2 = open(path2, 'rb')
            buf2 = data2.read()
            hasher2 = hashlib.md5()
            temp = hasher2.update(buf2)
            temp_md52 = hasher2.hexdigest()
            if temp_md51 == temp_md52:
                img_list.pop(i)
                break
    print(len(img_list))
    for sample in img_list:
        path = "/data/ai_lane/trainval_tag/" + sample
        img = cv2.imread(path[:-3]+"png",-1)
        id_set = set(img.reshape(-1))
        for id in id_set:
            class_ids[id]+=1
    print(class_ids)


