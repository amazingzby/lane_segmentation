import numpy as np

# char = ""
# npy_dir = "../img_class/train%s/"%char
# src_txt = "/data/ai_lane/train_list%s.txt"%char
# dst_txt = "/data/ai_lane/train_list%s_expand_v3.txt"%char
npy_dir = "../img_class/trainval/"
src_txt = "/data/ai_lane/trainval_list.txt"
dst_txt = "/data/ai_lane/trainval_list_balance.txt"

if __name__ == "__main__":
    train_list = []
    file_list = np.loadtxt(src_txt,delimiter=" ",dtype=str)
    num_dst = 1000
    print(file_list.shape)
    for idx in range(1,20):
        npy_data = np.load(npy_dir + str(idx) + ".npy")
        num_imgs = len(npy_data)
        if num_imgs < num_dst:
            repeat = (num_dst // num_imgs)
            repeat = min(repeat,25)
            for sample in npy_data:
                img_path = 'trainval_pic/%s.jpg' % (sample)
                label_path = 'trainval_tag/%s.png' % (sample)
                for _ in range(repeat):
                    file_list = np.concatenate([file_list, [[img_path, label_path]]], axis=0)
    for _ in range(5):
        np.random.shuffle(file_list)
    f = open(dst_txt,"w")
    for sample in file_list:
        f.write(sample[0]+" "+sample[1]+"\n")
    print(len(file_list))
    # for idx in [5,14]:
    #     npy_data = np.load(npy_dir+str(idx)+".npy")
    #     for sample in npy_data:
    #         img_path = 'trainval_pic_crop/%s.jpg'%(sample)
    #         label_path = 'trainval_tag_crop/%s.png'%(sample)
    #         for _ in range(5):
    #             file_list = np.concatenate([file_list,[[img_path,label_path]]],axis=0)
    # for idx in [11,17,18]:
    #     npy_data = np.load(npy_dir+str(idx)+".npy")
    #     for sample in npy_data:
    #         img_path = 'trainval_pic_crop/%s.jpg'%(sample)
    #         label_path = 'trainval_tag_crop/%s.png'%(sample)
    #         for _ in range(20):
    #             file_list = np.concatenate([file_list,[[img_path,label_path]]],axis=0)
    #
    # for _ in range(5):
    #     np.random.shuffle(file_list)
    # f = open(dst_txt,"w")
    # for sample in file_list:
    #     f.write(sample[0]+" "+sample[1]+"\n")