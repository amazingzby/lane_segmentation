import os
import sys
import paddle
import numpy as np
import paddle.fluid as fluid
from paddle.fluid import profiler
from models.hrnet import hrnet
from models.deeplab import deeplabv3p
from models.ocrnet import ocrnet
from models.loss import multi_softmax_with_loss
from models.lovasz_losses import multi_lovasz_softmax_loss

from data.reader import ModelPhase,SegDataset
from config import lane_cfg as cfg
from utils.timer import Timer,calculate_eta
from eval import evaluate
from load_model_utils import load_pretrained_weights

train_list = cfg["train_list"]
data_dir = cfg["data_dir"]
savd_dir = cfg["savd_dir"]
num_gpus = cfg["num_gpus"]
begin_epoch = cfg["begin_epoch"]
log_steps = 10
num_epochs = cfg["num_epochs"]
snapshot = cfg["snapshot"]
#模型输入
total_train_images = cfg["train_images"]
batch_size = cfg["batch_size"]
train_shape = cfg["dst_shape"]
all_step = (total_train_images // batch_size) * (num_epochs - begin_epoch + 1)

def build_model(main_prog, start_prog, phase=ModelPhase.TRAIN):
    width = train_shape[0]
    height = train_shape[1]
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
                              capacity=256,iterable=False,
                              use_double_buffer=True)
            net_name = cfg["model"]
            if net_name == "deeplab":
                net = deeplabv3p
            elif net_name == "hrnet":
                net = hrnet
            elif net_name == "ocrnet":
                net = ocrnet
            #logit = hrnet(image,class_num)
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
            if ModelPhase.is_eval(phase):
                return data_loader,avg_loss,pred,label,mask
            if ModelPhase.is_train(phase):
                decay_step = all_step
                power = 0.9

                #decay_lr = fluid.layers.piecewise_decay(cfg["decay_step"],values=cfg["decay_values"])
                #optimizer = fluid.optimizer.Momentum(learning_rate=decay_lr,momentum=0.9,
                #            regularization=fluid.regularizer.L2Decay(regularization_coeff=4e-05))
                poly_lr = fluid.layers.polynomial_decay(cfg["lr"],decay_step,end_learning_rate=0,power=power)
                optimizer  = fluid.optimizer.Adam(learning_rate=poly_lr,beta1=0.9,beta2=0.99,
                             regularization=fluid.regularizer.L2Decay(regularization_coeff=4e-05))

                optimizer.minimize(avg_loss)
                return data_loader,avg_loss,poly_lr,pred,label,mask

def save_infer_program(test_program, ckpt_dir):
    _test_program = test_program.clone()
    _test_program.desc.flush()
    _test_program.desc._set_version()
    paddle.fluid.core.save_op_compatible_info(_test_program.desc)
    with open(os.path.join(ckpt_dir, 'model') + ".pdmodel", "wb") as f:
        f.write(_test_program.desc.serialize_to_string())

def train():
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    dataset = SegDataset(file_list=train_list,mode=ModelPhase.TRAIN,
                         shuffle=True,data_dir=data_dir)

    def data_generator():
        data_gen = dataset.multiprocess_generator(
                   num_processes=8,max_queue_size=256)
        #data_gen = dataset.generator()
        batch_data = []
        for b in data_gen:
            batch_data.append(b)
            if len(batch_data) == batch_size:
                for item in batch_data:
                    yield item[0],item[1],item[2]
                batch_data = []
    #GPU
    place = fluid.CUDAPlace(0)
    places = fluid.cuda_places()
    dev_count = num_gpus #4GPU

    batch_size_per_dev = batch_size//dev_count
    print("batch_size_per_dev: {}".format(batch_size_per_dev))

    # build model
    data_loader,avg_loss,lr,pred,grts,masks = build_model(train_prog,
                startup_prog,phase=ModelPhase.TRAIN)
    build_model(test_prog,fluid.Program(),phase=ModelPhase.EVAL)
    data_loader.set_sample_generator(data_generator,batch_size=batch_size_per_dev,
                                     drop_last=True)

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = fluid.core.get_cuda_device_count()
    exec_strategy.num_iteration_per_drop_scope = 100
    build_strategy = fluid.BuildStrategy()

    print("Sync BatchNorm strategy is effective.")
    build_strategy.sync_batch_norm = True
    compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name=avg_loss.name,exec_strategy=exec_strategy,build_strategy=build_strategy)
    print("Load pretrained model for %s"%cfg["pretrained"])
    load_pretrained_weights(exe, train_prog,cfg["pretrained"])

    fetch_list = [avg_loss.name,lr.name]
    step = 0
    avg_loss = 0.0
    best_mIoU = 0.0
    timer = Timer()
    timer.start()
    for epoch in range(begin_epoch,num_epochs+1):
        data_loader.start()
        while True:
            try:
                loss,lr = exe.run(program=compiled_train_prog,fetch_list=fetch_list,return_numpy=True)
                #avg_loss += np.mean(np.array(loss))
                avg_loss += np.mean(np.array(loss))
                step += 1
                if step % 10 == 0:
                    speed = log_steps / timer.elapsed_time()
                    avg_loss /= log_steps
                    print("epoch={} step={} lr={:.5f} loss={:.4f} step/sec={:.3f} | ETA {}".format(
                        epoch,step,lr[0],avg_loss,speed,calculate_eta(all_step,speed)
                    ))
                    sys.stdout.flush()
                    avg_loss = 0.0
                    timer.restart()

            except fluid.core.EOFException:
                data_loader.reset()
                break
            except Exception as e:
                print(e)
        if epoch%snapshot == 0 or epoch == num_epochs:
            ckpt_dir = os.path.join(savd_dir,str(epoch))
            if not os.path.isdir(ckpt_dir):
                os.mkdir(ckpt_dir)
            print("Save model checkpoint to {}".format(ckpt_dir))
            fluid.save(train_prog,os.path.join(ckpt_dir,'model'))
            save_infer_program(test_prog,ckpt_dir)
            #print("Evaluation start")
            #_, mean_iou, _, mean_acc = evaluate(
            #                ckpt_dir=ckpt_dir)
            #if mean_iou > best_mIoU:
            #    best_mIoU = mean_iou
            #    print("best model epoch:{},mIoU = {:.4f}".format(epoch,mean_iou))
if __name__ == "__main__":
    train()

