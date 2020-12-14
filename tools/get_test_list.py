import os
import cv2
#测试集
data_dir = "../dataset"
test_dir = "testB"
f1_test = open(os.path.join(data_dir,"test_list_partA.txt"),"w")
f2_test = open(os.path.join(data_dir,"test_list_partB.txt"),"w")
for img_name in os.listdir(os.path.join(data_dir,test_dir)):
    path = os.path.join(test_dir,img_name)
    img = cv2.imread(os.path.join(data_dir,path),-1)
    if img.shape[0]==1080 and img.shape[1]==1920:
        f1_test.write(path+"\n")
    elif img.shape[0]==720 and img.shape[1]==1280:
        f2_test.write(path+"\n")
    else:
        print("Wrong sample!")
        exit(0)
f1_test.close()
f2_test.close()
