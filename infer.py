import os
import cv2
import numpy as np
from models.deeplab import deeplabv3p
from models.hrnet import hrnet
from models.ocrnet import ocrnet
import paddle.fluid as fluid
from config import lane_cfg as cfg
from data.reader import SegDataset,ModelPhase

infer_shape = cfg["dst_shape"]
valid_shape = cfg["src_shape"]
pad_shape = cfg["zero_pads"]

def softmax(logit):
    logit = fluid.layers.transpose(logit, [0, 2, 3, 1])
    logit = fluid.layers.softmax(logit)
    logit = fluid.layers.transpose(logit, [0, 3, 1, 2])
    return logit

def build_model(main_prog, start_prog, phase=ModelPhase.TRAIN):
    width = infer_shape[0]
    height= infer_shape[1]
    image_shape = [-1,3,height,width]
    grt_shape = [-1,1,height,width]
    class_num = 20
    with fluid.program_guard(main_prog, start_prog):
        with fluid.unique_name.guard():
            image = fluid.data(name='image', shape=image_shape, dtype='float32')
            label = fluid.data(name='label', shape=grt_shape, dtype='int32')
            mask  = fluid.data(name='mask', shape=grt_shape, dtype='int32')

            if ModelPhase.is_train(phase) or ModelPhase.is_eval(phase):
                data_loader = fluid.io.DataLoader.from_generator(
                              feed_list=[image, label, mask],
                              capacity=64,iterable=False,
                              use_double_buffer=True)
            net_name = cfg["model"]
            if net_name == "deeplab":
                net = deeplabv3p
            elif net_name == "hrnet":
                net = hrnet
            elif net_name == "ocrnet":
                net = ocrnet
            logits = net(image,class_num)

            if isinstance(logits, tuple):
                logit = logits[0]
            else:
                logit = logits

            if logit.shape[2:] != label.shape[2:]:
                logit = fluid.layers.resize_bilinear(logit, label.shape[2:])
            out = fluid.layers.transpose(logit,[0,2,3,1])
            pred = fluid.layers.argmax(out, axis=3)
            pred = fluid.layers.unsqueeze(pred, axes=[3])
            logit = softmax(logit)
            return pred,logit

def inference():
    vis_file_list = cfg["test_list"]
    dataset = SegDataset(file_list=vis_file_list,
                         mode=ModelPhase.VISUAL,
                         data_dir = cfg["data_dir"])
    startup_prog = fluid.Program()
    test_prog = fluid.Program()
    pred,logit = build_model(test_prog, startup_prog, phase=ModelPhase.VISUAL)
    test_prog = test_prog.clone(for_test=True)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    ckpt_dir = cfg["infer_model"]
    print("Load model form %s"%ckpt_dir)
    fluid.load(test_prog, os.path.join(ckpt_dir, 'model'), exe)

    fetch_list = [pred.name,logit.name]
    test_reader = dataset.batch(dataset.generator,batch_size=1,is_test=True)
    img_cnt = 0
    for imgs,grts,img_names,org_shapes in test_reader:
        img_cnt += 1
        #pred_shape = (imgs.shape[2], imgs.shape[3])
        pred,logits = exe.run(program=test_prog,feed={'image': imgs},
                              fetch_list=fetch_list,return_numpy=True)
        num_imgs = pred.shape[0]
        for i in range(num_imgs):
            print("Process %d:%s" % (img_cnt,img_names[i]))
            res_map = np.squeeze(pred[i, :, :, :]).astype(np.uint8)
            res_npy = np.squeeze(logits).transpose(1,2,0)
            p_w,p_h = pad_shape
            width,height = valid_shape
            #res_map = res_map[p_h:(p_h+height),p_w:(p_w+width)]
            #res_npy = res_npy[p_h:(p_h+height),p_w:(p_w+width)]
            org_shape = (org_shapes[i][0], org_shapes[i][1])
            res_map = cv2.resize(
                res_map, (org_shape[1], org_shape[0]),
                interpolation=cv2.INTER_NEAREST)
            res_npy = cv2.resize(
                res_npy,(org_shape[1], org_shape[0]),
                interpolation=cv2.INTER_NEAREST)
            img_id = img_names[i].split("/")[-1][:-4]
            res_npy = res_npy.astype(np.float16)
            cv2.imwrite(os.path.join(cfg["result_dir"],img_id+".png"),res_map)
            np.save(os.path.join(cfg["result_dir"],img_id+".npy"),res_npy)




if __name__ == "__main__":
    inference()
