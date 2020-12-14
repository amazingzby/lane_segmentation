import numpy as np
import cv2

data_dir = "/data/ai_lane/"
npy_path = "../img_class/trainval/"

#注：18类只保留10009052
#17有栅栏-伸出来左实右虚 keep id
#10009821,10009925,10010240,10010448,10010481,10010967,10011032,10015074,10016908,10020359
#16删除 10008532,10013940
#14删除
#10008415,10008645,10009187,10009816,10010078,10010775,10012030,10012200,10012409,10015005,10016917,10018268,10018268,
#10018831,10019889,10020519,10021864,

if __name__ == "__main__":
    class_id = "14"
    npy_data = np.load(npy_path+class_id+".npy")
    for img_id in npy_data:
        img = cv2.imread(data_dir+"trainval_pic/"+img_id+".jpg",-1)
        tag = cv2.imread(data_dir+"trainval_tag/"+img_id+".png",-1)
        id_int = int(class_id)
        img_a = img.copy()
        img_a[tag==id_int] = [0,0,255]
        img_saved = np.concatenate([img,img_a],axis=1)
        cv2.imwrite("images/"+class_id+"_"+img_id+".jpg",img_saved)

