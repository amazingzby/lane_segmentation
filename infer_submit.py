from models.ocrnet import ocrnet
import os
import cv2
import sys
from datetime import datetime
import time
import numpy as np
import paddle.fluid as fluid

def normalize_image(img):
    """ 像素归一化后减均值除方差 """
    mean = [0.5, 0.5, 0.5]
    std  = [0.5, 0.5, 0.5]
    img = img.transpose((2, 0, 1)).astype('float32') / 255.0
    img_mean = np.array(mean).reshape(len(mean), 1, 1)
    img_std = np.array(std).reshape(len(std), 1, 1)
    img -= img_mean
    img /= img_std
    return img

def get_input_img(data_dir,img_name,img_shape):
    img_path = os.path.join(data_dir,img_name)
    img = cv2.imread(img_path,-1)
    img = cv2.resize(img,(img_shape[0],img_shape[1]), interpolation=cv2.INTER_LINEAR)
    img = img[..., ::-1]
    img = normalize_image(img)
    return img

def get_batch_data(data_dir,samples_name,img_shape):
    batch_imgs = []
    for sample_name in samples_name:
        img = get_input_img(data_dir,sample_name,img_shape)
        batch_imgs.append(img)
    return np.array(batch_imgs)

def softmax(logit):
    logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
    logit = fluid.layers.softmax(logit)
    logit = fluid.layers.transpose(logit, [0, 3, 1, 2])
    return logit

def build_model(main_prog, start_prog,infer_shape):
    width = infer_shape[0]
    height= infer_shape[1]
    image_shape = [-1,3,height,width]
    class_num = 20
    with fluid.program_guard(main_prog, start_prog):
        with fluid.unique_name.guard():
            image = fluid.data(name='image', shape=image_shape, dtype='float32')
            net = ocrnet
            logits = net(image,class_num)
            logit = logits[0]
            out = fluid.layers.transpose(logit,[0,2,3,1])
            pred = fluid.layers.argmax(out, axis=3)
            pred = fluid.layers.unsqueeze(pred, axes=[3])
            logit = softmax(logit)
            return pred,logit

def infer():
    data_dir   = "dataset"
    test_path1 ="dataset/test_list_partA.txt"
    test_path2 ="dataset/test_list_partB.txt"
    filenames1=np.loadtxt(test_path1,dtype=str)
    filenames2=np.loadtxt(test_path2, dtype=str)
    batch_size = 8
    infer_shapes = [[320,180],[480,540],[640,360],[640,720],[960,540],[960,1080],[1280,720]]
    epochs = [14,12,13,13,14,12,12]
    #num_resolution = len(infer_shapes)
    num_resolution = 7
    startup_prog_list,test_prog_list,pred_list,logit_list = [],[],[],[]
    for idx in range(num_resolution):
        startup_prog_list.append(fluid.Program())
        test_prog = fluid.Program()
        pred, logit = build_model(test_prog, startup_prog_list[idx],infer_shapes[idx])
        test_prog_list.append(test_prog.clone(for_test=True))
        pred_list.append(pred)
        logit_list.append(logit)
    
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    iterations1 = len(filenames1)//batch_size
    if len(filenames1)%batch_size:
        iterations1 += 1
    iterations2 = len(filenames2)//batch_size
    if len(filenames2)%batch_size:
        iterations2 += 1
    num_iter = 0
    total_iter = iterations1 + iterations2
    #数据集1：1920*1080
    for iter in range(iterations1):
        start = iter*batch_size
        end = min((iter+1)*batch_size,len(filenames1))
        samples_name = filenames1[start:end]
        samples_res = []
        for idx in range(num_resolution):
            ckpt_dir = "saved_model/ocrnet_"+str(infer_shapes[idx][0])+"_"+str(infer_shapes[idx][1])+"_epoch_"+str(epochs[idx])
            fluid.load(test_prog_list[idx], os.path.join(ckpt_dir, 'model'), exe)
            samples_data = get_batch_data(data_dir,samples_name,infer_shapes[idx])
            _,logits = exe.run(program=test_prog_list[idx], feed={'image': samples_data},
                                   fetch_list=[pred_list[idx].name,logit_list[idx].name], return_numpy=True) 
            res_npy = logits.transpose(2, 3, 1, 0)
            res_npy = res_npy.reshape(res_npy.shape[0],res_npy.shape[1],(res_npy.shape[2]*res_npy.shape[3]))
            res_npy = cv2.resize(res_npy,(1920,1080),interpolation=cv2.INTER_NEAREST)
            res_npy = res_npy.reshape(1080,1920,20,len(samples_name))
            if len(samples_res) > 1:
                samples_res += res_npy
            else:
                samples_res = res_npy
        samples_label = np.argmax(samples_res,axis=2).astype(np.uint8)
        for idx in range(len(samples_name)):
            img_name = "result/result_img/"+samples_name[idx].split("/")[-1][:-4]+".png"
            img_data = samples_label[:,:,idx]
            cv2.imwrite(img_name,img_data)
        num_iter += 1
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),end=" ")
        print("Processing %d/%d iterations"%(num_iter,total_iter))
        sys.stdout.flush()


    #数据集2：1280*720
    for iter in range(iterations2):
        start = iter*batch_size
        end = min((iter+1)*batch_size,len(filenames2))
        samples_name = filenames2[start:end]
        samples_res = np.zeros((720, 1280, 20, len(samples_name)))
        for idx in range(num_resolution):
            ckpt_dir = "saved_model/ocrnet_"+str(infer_shapes[idx][0])+"_"+str(infer_shapes[idx][1])+"_epoch_"+str(epochs[idx])
            fluid.load(test_prog_list[idx], os.path.join(ckpt_dir, 'model'), exe)
            samples_data = get_batch_data(data_dir,samples_name,infer_shapes[idx])
            logits = exe.run(program=test_prog_list[idx], feed={'image': samples_data},
                                   fetch_list=[logit_list[idx].name], return_numpy=True)
            res_npy = logits[0].transpose(2, 3, 1, 0)
            res_npy = res_npy.reshape(res_npy.shape[0],res_npy.shape[1],(res_npy.shape[2]*res_npy.shape[3]))
            res_npy = cv2.resize(res_npy,(1280,720),interpolation=cv2.INTER_NEAREST)
            res_npy = res_npy.reshape(720,1280,20,len(samples_name))
            samples_res += res_npy
        samples_label = np.argmax(samples_res,axis=2).astype(np.uint8)
        for idx in range(len(samples_name)):
            img_name = "result/result_img/"+samples_name[idx].split("/")[-1][:-4]+".png"
            img_data = samples_label[:,:,idx]
            cv2.imwrite(img_name,img_data)
        num_iter += 1
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),end=" ")
        print("Processing %d/%d iterations"%(num_iter,total_iter))
        sys.stdout.flush()


if __name__ == "__main__":
    infer()

