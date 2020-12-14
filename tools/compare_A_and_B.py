#比较两个数据集是否有相同id
import os
from glob import glob
filenames1 = list(glob("/data/ai_lane/train_tag/*.png"))
filenames2 = list(glob("/data/ai_lane/PreliminaryData/train_label/*.png"))
filenames = filenames1 + filenames2
img_dir = "/data/ai_lane/trainval_tag"
if __name__ == "__main__":
    for img_path in filenames:
        cmd = "cp "+ img_path + " "+ img_dir
        print(cmd)
        os.system(cmd)
