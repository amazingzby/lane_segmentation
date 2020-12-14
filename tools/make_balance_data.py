import numpy as np

npy_dir = "img_class/train/"
src_txt = "/data/ai_lane/train_list.txt"
dst_txt = "/data/ai_lane/train_list_expand_v2.txt"


if __name__ == "__main__":
    train_list = []
    file_list = np.loadtxt(src_txt,delimiter=" ",dtype=str)
    print(file_list.shape)
    for idx in range(1,20):
        npy_data = np.load(npy_dir + str(idx) + ".npy")
        num_imgs = len(npy_data)
        if num_imgs < 1000:
            repeat = (1000 // num_imgs)
            for sample in npy_data:
                img_path = 'trainval_pic_crop/%s.jpg' % (sample)
                label_path = 'trainval_tag_crop/%s.png' % (sample)
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