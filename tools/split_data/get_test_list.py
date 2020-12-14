import os
#测试集
data_dir = "/data/ai_lane"
test_dir = "testA_crop"
f_test = open(os.path.join(data_dir,"test_list.txt"),"w")
for img_name in os.listdir(os.path.join(data_dir,test_dir)):
    path = os.path.join(test_dir,img_name)
    f_test.write(path+"\n")
f_test.close()