import numpy as np

if __name__ == "__main__":
    select_id = "11"
    trainval_npy = np.load("trainval/"+select_id+".npy")
    val_npy = np.load("val/"+select_id+".npy")
    np.random.shuffle(trainval_npy)
    print(trainval_npy)