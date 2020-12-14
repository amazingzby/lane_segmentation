import os
import numpy as np

data_dir = "/data/ai_lane"

if __name__ == "__main__":
    f_train = open(os.path.join(data_dir, "train_list.txt"))
    f_eval = open(os.path.join(data_dir, "val_list.txt"))
    data_train = f_train.readlines()
    dataA1 = np.array(data_train)
    np.random.shuffle(dataA1)
    data_eval = f_eval.readlines()
    dataA2 = np.array(data_eval)
    num_A1 = len(dataA1)
    num_A2 = len(dataA2)
    print(num_A1)
    print(num_A2)

    f_B1 = open(os.path.join(data_dir, "train_list_B.txt"),"w")
    f_B2 = open(os.path.join(data_dir, "val_list_B.txt"),"w")
    for idx in range(num_A2):
        f_B2.write(dataA1[idx])
    for idx in range(num_A2,num_A1):
        f_B1.write(dataA1[idx])
    for sample in dataA2:
        f_B1.write(sample)

    f_C1 = open(os.path.join(data_dir, "train_list_C.txt"),"w")
    f_C2 = open(os.path.join(data_dir, "val_list_C.txt"),"w")
    for idx in range(num_A2,num_A2*2):
        f_C2.write(dataA1[idx])
    for idx in range(num_A2):
        f_C1.write(dataA1[idx])
    for idx in range(num_A2*2,num_A1):
        f_C1.write(dataA1[idx])
    for sample in dataA2:
        f_C1.write(sample)


