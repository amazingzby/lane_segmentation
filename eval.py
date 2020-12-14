import os
import sys
import argparse
import pprint
import numpy as np
import paddle.fluid as fluid
from config import lane_cfg as cfg
from models.hrnet import hrnet
from models.deeplab import deeplabv3p
from models.ocrnet import ocrnet
from data.reader import ModelPhase,SegDataset
from models.loss import multi_softmax_with_loss
from models.lovasz_losses import multi_lovasz_softmax_loss
from utils.timer import Timer, calculate_eta
from metrics import ConfusionMatrix

eval_shape = cfg["dst_shape"]
def build_model(main_prog, start_prog, phase=ModelPhase.TRAIN):
    width = eval_shape[0]
    height = eval_shape[1]
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
            #logit = hrnet(image,class_num)
            net_name = cfg["model"]
            if net_name == "deeplab":
                net = deeplabv3p
            elif net_name == "hrnet":
                net = hrnet
            elif net_name == "ocrnet":
                net = ocrnet
            logits = net(image,class_num)
            if ModelPhase.is_train(phase) or ModelPhase.is_eval(phase):
                weight_softmax = [0.5] + ([5.0] * 19)
                softmax_loss = multi_softmax_with_loss(logits,label,mask,class_num,weight_softmax)
                if cfg["lovasz_loss"]:
                    lovasz_loss =  multi_lovasz_softmax_loss(logits,label,mask)
                    lovasz_weight = cfg["lovasz_weight"]
                    softmax_weight = 1 - lovasz_weight
                    avg_loss = (lovasz_weight * lovasz_loss) + (softmax_weight * softmax_loss)
                else:
                    avg_loss = softmax_loss

            if isinstance(logits, tuple):
                logit = logits[0]
            else:
                logit = logits
            if logit.shape[2:] != label.shape[2:]:
                logit = fluid.layers.resize_bilinear(logit, label.shape[2:])
            if ModelPhase.is_predict(phase):
                logit = softmax(logit)
            out = fluid.layers.transpose(logit,[0,2,3,1])
            pred = fluid.layers.argmax(out, axis=3)
            pred = fluid.layers.unsqueeze(pred, axes=[3])
            return data_loader,avg_loss,pred,label,mask

def evaluate(ckpt_dir=None):
    np.set_printoptions(precision=5, suppress=True)
    startup_prog = fluid.Program()
    test_prog = fluid.Program()
    dataset = SegDataset(
        file_list=cfg["val_list"],
        mode=ModelPhase.EVAL,
        data_dir=cfg["data_dir"])
    def data_generator():
        data_gen = dataset.generator()
        for b in data_gen:
            yield b[0],b[1],b[2]

    data_loader, avg_loss, pred, grts, masks = build_model(
        test_prog, startup_prog, phase=ModelPhase.EVAL)
    data_loader.set_sample_generator(data_generator, drop_last=False, batch_size=cfg["batch_size"])

    places = fluid.cuda_places()
    place = places[0]
    dev_count = len(places)
    print("#Device count: {}".format(dev_count))
    exe = fluid.Executor(place)
    exe.run(startup_prog)
    test_prog = test_prog.clone(for_test=True)
    fluid.load(test_prog, os.path.join(ckpt_dir, 'model'), exe)
    #fluid.io.load_params(exe, ckpt_dir, main_program=test_prog)
    np.set_printoptions(precision=4, suppress=True, linewidth=160, floatmode="fixed")
    conf_mat = ConfusionMatrix(20,streaming=True)
    fetch_list = [avg_loss.name, pred.name, grts.name, masks.name]
    num_images = 0
    step = 0
    all_step = cfg["test_images"] // cfg["batch_size"] + 1
    timer = Timer()
    timer.start()
    data_loader.start()
    while True:
        try:
            step += 1
            loss, pred, grts, masks = exe.run(
                test_prog, fetch_list=fetch_list, return_numpy=True)
            loss = np.mean(np.array(loss))
            num_images += pred.shape[0]
            conf_mat.calculate(pred, grts, masks)
            _, iou = conf_mat.mean_iou()
            _, acc = conf_mat.accuracy()
            speed = 1.0 / timer.elapsed_time()
            print(
                "[EVAL]step={} loss={:.5f} acc={:.4f} IoU={:.4f} step/sec={:.2f} | ETA {}"
                .format(step, loss, acc, iou, speed,
                calculate_eta(all_step - step, speed)))
            timer.restart()
            sys.stdout.flush()
        except fluid.core.EOFException:
            break
    category_iou, avg_iou = conf_mat.mean_iou()
    category_acc, avg_acc = conf_mat.accuracy()
    print("[EVAL]#image={} acc={:.4f} IoU={:.4f}".format(
        num_images, avg_acc, avg_iou))
    print("[EVAL]Category IoU:", category_iou)
    print("[EVAL]Category Acc:", category_acc)
    print("[EVAL]Kappa:{:.4f}".format(conf_mat.kappa()))

    return category_iou, avg_iou, category_acc, avg_acc

if __name__ == "__main__":
    evaluate(ckpt_dir=cfg["infer_model"])

