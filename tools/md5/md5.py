import hashlib
import glob
import numpy as np

if __name__ == "__main__":
    md51 = np.load("md51.npy")
    md52 = np.load("md52.npy")
    md53 = np.load("md53.npy")
    md5_train = np.concatenate([md51,md52])
    md5_test = md53
    f_train = open("same_train.txt", "w")
    for i in range(len(md5_train)-1):
        for j in range(i+1,len(md5_train)):
            if md5_train[i,1] == md5_train[j,1]:
                img_id1 = md5_train[i,0].split("/")[-1]
                img_id2 = md5_train[j,0].split("/")[-1]
                f_train.write(img_id1+" "+img_id2+"\n")
    f_train.close()
    f_test = open("same_test_train.txt", "w")
    for i in range(len(md5_train)):
        for j in range(len(md5_test)):
            if md5_train[i,1] == md5_test[j,1]:
                img_id1 = md5_train[i,0].split("/")[-1]
                img_id2 = md5_test[j,0].split("/")[-1]
                f_test.write(img_id1+" "+img_id2+"\n")
    f_test.close()

    # for m2 in md51:
    #     for m3 in md53:
    #         if m2[1] == m3[1]:
    #             same_count += 1
    #             print(m2[0],end=" ")
    #             print(m3[0])
    # print(same_count)
    #2和3有14张重复
    # filenames1 = glob.glob("/home/zby/data/dataset/ai_lane/train_pic/*.jpg")
    # filenames2 = glob.glob("/home/zby/data/dataset/ai_lane/PreliminaryData/train/*.jpg")
    # filenames3 = glob.glob("/home/zby/data/dataset/ai_lane/PreliminaryData/testA/*.jpg")
    # print(len(filenames1))
    # print(len(filenames2))
    # print(len(filenames3))
    # md51 = []
    # md52 = []
    # md53 = []
    # for file in filenames1:
    #         data = open(file,'rb')
    #         buf = data.read()
    #         hasher = hashlib.md5()
    #         temp = hasher.update(buf)
    #         temp_md5 = hasher.hexdigest()
    #         md51.append([file.split("/")[-1],temp_md5])
    # print("part 1 done!")
    # for file in filenames2:
    #         data = open(file,'rb')
    #         buf = data.read()
    #         hasher = hashlib.md5()
    #         temp = hasher.update(buf)
    #         temp_md5 = hasher.hexdigest()
    #         md52.append([file.split("/")[-1],temp_md5])
    # print("part 2 done!")
    # for file in filenames3:
    #         data = open(file,'rb')
    #         buf = data.read()
    #         hasher = hashlib.md5()
    #         temp = hasher.update(buf)
    #         temp_md5 = hasher.hexdigest()
    #         md53.append([file.split("/")[-1],temp_md5])
    # print("part 3 done!")
    # md51 = np.array(md51)
    # md52 = np.array(md52)
    # md53 = np.array(md53)
    # np.save("md51.npy",md51)
    # np.save("md52.npy",md52)
    # np.save("md53.npy",md53)
    # print("md5 process done!")
    # for m2 in md52:
    #     for m1 in md51:
    #         if m2 == m1:
    #             print("helloworld")