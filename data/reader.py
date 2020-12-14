import sys
import os
import math
import random
import functools
import io
import time
import codecs

import numpy as np
import paddle
import paddle.fluid as fluid
import cv2
from PIL import Image

import data.data_aug as aug
from data.data_aug import ModelPhase
from config import lane_cfg as cfg
from data.data_utils import GeneratorEnqueuer
import copy

class SegDataset(object):
    def __init__(self,
                 file_list,
                 data_dir,
                 shuffle=False,
                 mode=ModelPhase.TRAIN):
        self.mode = mode
        self.shuffle = shuffle
        self.data_dir = data_dir

        # NOTE: Please ensure file list was save in UTF-8 coding format
        with open(file_list) as flist:
            self.lines = [line.strip() for line in flist]
            self.all_lines = copy.deepcopy(self.lines)
            if shuffle:
                np.random.shuffle(self.lines)

    def generator(self):
        if self.shuffle:
            np.random.shuffle(self.lines)

        for line in self.lines:
            yield self.process_image(line, self.data_dir, self.mode)

    def sharding_generator(self, pid=0, num_processes=1):
        """
        Use line id as shard key for multiprocess io
        It's a normal generator if pid=0, num_processes=1
        """
        for index, line in enumerate(self.lines):
            # Use index and pid to shard file list
            if index % num_processes == pid:
                yield self.process_image(line, self.data_dir, self.mode)

    def batch_reader(self, batch_size):
        br = self.batch(self.reader, batch_size)
        for batch in br:
            yield batch[0], batch[1], batch[2]

    def multiprocess_generator(self, max_queue_size=32, num_processes=8):
        # Re-shuffle file list
        np.random.shuffle(self.lines)

        # Create multiple sharding generators according to num_processes for multiple processes
        generators = []
        for pid in range(num_processes):
            generators.append(self.sharding_generator(pid, num_processes))

        try:
            enqueuer = GeneratorEnqueuer(generators)
            enqueuer.start(max_queue_size=max_queue_size, workers=num_processes)
            while True:
                generator_out = None
                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        generator_out = enqueuer.queue.get(timeout=5)
                        break
                    else:
                        time.sleep(0.01)
                if generator_out is None:
                    break
                yield generator_out
        finally:
            if enqueuer is not None:
                enqueuer.stop()

    def batch(self, reader, batch_size, is_test=False, drop_last=False):
        def batch_reader(is_test=False, drop_last=drop_last):
            if is_test:
                imgs, grts, img_names, valid_shapes, org_shapes = [], [], [], [], []
                #for img, grt, img_name, valid_shape, org_shape in reader():
                for img,grt,img_name,org_shape in reader():
                    imgs.append(img)
                    grts.append(grt)
                    img_names.append(img_name)
                    #valid_shapes.append(valid_shape)
                    org_shapes.append(org_shape)
                    if len(imgs) == batch_size:
                        yield np.array(imgs), np.array(grts), img_names,org_shapes
                        imgs, grts, img_names,org_shapes = [], [], [],[]

                if not drop_last and len(imgs) > 0:
                    yield np.array(imgs), np.array(grts), img_names, np.array(
                        valid_shapes), np.array(org_shapes)
            else:
                imgs, labs, ignore = [], [], []
                bs = 0
                for img, lab, ig in reader():
                    imgs.append(img)
                    labs.append(lab)
                    ignore.append(ig)
                    bs += 1
                    if bs == batch_size:
                        yield np.array(imgs), np.array(labs), np.array(ignore)
                        bs = 0
                        imgs, labs, ignore = [], [], []

                if not drop_last and bs > 0:
                    yield np.array(imgs), np.array(labs), np.array(ignore)

        return batch_reader(is_test, drop_last)

    #返回图像，label,图像名，label名 img, grt, img_name, grt_name
    def load_image(self, line, src_dir, mode=ModelPhase.TRAIN):
        # original image cv2.imread flag setting
        parts = line.strip().split(' ')
        if mode == ModelPhase.VISUAL:
            img_name, grt_name = parts[0], None
        else:
            img_name, grt_name = parts[0], parts[1]

        img_path = os.path.join(src_dir, img_name)
        img = cv2.imread(img_path, -1)

        if grt_name is not None:
            grt_path = os.path.join(src_dir, grt_name)
            grt = cv2.imread(grt_path,-1)
        else:
            grt = None

        if img is None:
            raise Exception(
                "Empty image, source image path: {}".format(img_path))

        img_height = img.shape[0]
        img_width = img.shape[1]

        if grt is not None:
            grt_height = grt.shape[0]
            grt_width = grt.shape[1]

            if img_height != grt_height or img_width != grt_width:
                raise Exception(
                    "Source img and label img must has the same size.")
        else:
            if mode == ModelPhase.TRAIN or mode == ModelPhase.EVAL:
                raise Exception(
                    "No laber image path for image '{}' when training or evaluating. "
                    .format(img_path))
        return img, grt, img_name, grt_name

    def normalize_image(self, img):
        """ 像素归一化后减均值除方差 """
        img = img.transpose((2, 0, 1)).astype('float32') / 255.0
        img_mean = np.array(cfg["mean"]).reshape(len(cfg["mean"]), 1, 1)
        img_std = np.array(cfg["std"]).reshape(len(cfg["mean"]), 1, 1)
        img -= img_mean
        img /= img_std
        return img

    def process_image(self, line, data_dir, mode):
        """ process_image """
        img, grt, img_name, grt_name = self.load_image(
            line, data_dir, mode=mode)
        if mode == ModelPhase.TRAIN:
            img,grt = aug.compose(img,grt)
        elif ModelPhase.is_eval(mode):
            img,grt = aug.crop_eval(img, grt)
        elif ModelPhase.is_visual(mode):
            org_shape = [img.shape[0], img.shape[1]]
            img,grt = aug.crop_eval(img,grt)
        else:
            raise ValueError("Dataset mode={} Error!".format(mode))

        # Normalize image
        if cfg["to_rgb"]:
            img = img[..., ::-1]
        img = self.normalize_image(img)

        if ModelPhase.is_train(mode) or ModelPhase.is_eval(mode):
            grt = np.expand_dims(np.array(grt).astype('int32'), axis=0)
            ignore = (grt != 255).astype('int32')

        if ModelPhase.is_train(mode):
            return (img, grt,ignore)
        elif ModelPhase.is_eval(mode):
            return (img, grt,ignore)
        elif ModelPhase.is_visual(mode):
            return (img, grt, img_name,org_shape)#, valid_shape, org_shape)

if __name__ == "__main__":
    data = SegDataset("/data/ai_lane/train_list.txt","/data/ai_lane",shuffle=True,mode="eval")
    generator = data.generator()
    box_area = 640 * 256
    aspect_ratio = 640 / 256
    area_ratio_list = []
    aspect_ratio_list = []
    for i in range(10000):
        img,grt,ignore = next(generator)
        continue
        h,w = img.shape[:2]
        cur_area = h * w
        cur_ratio = cur_area / box_area
        cur_aspect= (w / h)
        area_ratio_list.append(cur_ratio)
        aspect_ratio_list.append(cur_aspect)
        print("image number:%d"%(i+1),end=",")
        print("current box ratio:%f ,current aspect ratio:%f"%(cur_ratio,cur_aspect))
        avg_box = np.mean(area_ratio_list)
        avg_asp = np.mean(aspect_ratio_list)
        #average box ratio:1.369730 average aspect ratio:0.804709
        #加aspect后average box ratio:1.576639 average aspect ratio:0.813890
        #average box ratio:1.570852 average aspect ratio:0.814922
        #average box ratio:2.268533 average aspect ratio:0.815557
        #更改计数方式
        #average box ratio:2.265420 average aspect ratio:2.041752
        #426.366*870.53
        print("     average box ratio:%f average aspect ratio:%f"%(avg_box,avg_asp))
