from __future__ import print_function
import cv2
import numpy as np
from config import lane_cfg as cfg

class ModelPhase(object):
    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'predict'
    VISUAL = 'visual'

    @staticmethod
    def is_train(phase):
        return phase == ModelPhase.TRAIN

    @staticmethod
    def is_predict(phase):
        return phase == ModelPhase.PREDICT

    @staticmethod
    def is_eval(phase):
        return phase == ModelPhase.EVAL

    @staticmethod
    def is_visual(phase):
        return phase == ModelPhase.VISUAL

    @staticmethod
    def is_valid_phase(phase):
        """ Check valid phase """
        if ModelPhase.is_train(phase) or ModelPhase.is_predict(phase) \
                or ModelPhase.is_eval(phase) or ModelPhase.is_visual(phase):
            return True
        return False

def saturation_jitter(cv_img, jitter_range):
    #饱和度调节
    greyMat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    greyMat = greyMat[:, :, None] * np.ones(3, dtype=int)[None, None, :]
    cv_img = cv_img.astype(np.float32)
    cv_img = cv_img * (1 - jitter_range) + jitter_range * greyMat
    cv_img = np.where(cv_img > 255, 255, cv_img)
    cv_img = cv_img.astype(np.uint8)
    return cv_img

def brightness_jitter(cv_img, jitter_range):
    cv_img = cv_img.astype(np.float32)
    cv_img = cv_img * (1.0 - jitter_range)
    cv_img = np.where(cv_img > 255, 255, cv_img)
    cv_img = cv_img.astype(np.uint8)
    return cv_img

def contrast_jitter(cv_img, jitter_range):
    greyMat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(greyMat)
    cv_img = cv_img.astype(np.float32)
    cv_img = cv_img * (1 - jitter_range) + jitter_range * mean
    cv_img = np.where(cv_img > 255, 255, cv_img)
    cv_img = cv_img.astype(np.uint8)
    return cv_img

def random_rotation(crop_img, crop_seg,do_rotation):
    h,w = crop_img.shape[:2]
    pc = (w // 2, h // 2)
    r = cv2.getRotationMatrix2D(pc, do_rotation, 1.0)
    cos = np.abs(r[0, 0])
    sin = np.abs(r[0, 1])

    #保证像素不丢失
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))

    (cx, cy) = pc
    r[0, 2] += (nw / 2) - cx
    r[1, 2] += (nh / 2) - cy
    dsize = (nw, nh)
    crop_img = cv2.warpAffine(
            crop_img,
            r,
            dsize=dsize,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0)
    crop_seg = cv2.warpAffine(
            crop_seg,
            r,
            dsize=dsize,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
    return crop_img, crop_seg

def random_aspect(crop_img,crop_seg,rich_crop_scale,rich_crop_aspect_ratio):
    h,w = crop_img.shape[:2]
    if (np.random.randint(10) < 5):
        h_expand = int(h*rich_crop_aspect_ratio)
        dh = np.random.randint(0,(h_expand - h + 1))
        crop_img_a = np.zeros((h_expand,w,3),dtype=crop_img.dtype)
        crop_img_a[:,:,:] = 127
        crop_seg_a = np.zeros((h_expand,w),dtype=crop_seg.dtype)
        crop_img_a[dh:(dh+h)] = crop_img
        crop_seg_a[dh:(dh+h)] = crop_seg
        #return crop_img_a,crop_seg_a
    else:
        w_expand = int(w * rich_crop_aspect_ratio)
        dw = np.random.randint(0,(w_expand - w + 1))
        crop_img_a = np.zeros((h,w_expand,3),dtype=crop_img.dtype)
        crop_img_a[:,:,:] = 127
        crop_seg_a = np.zeros((h,w_expand),dtype=crop_seg.dtype)
        crop_img_a[:,dw:(dw+w)] = crop_img
        crop_seg_a[:,dw:(dw+w)] = crop_seg
        #return crop_img_a,crop_seg_a
    crop_h,crop_w = crop_img_a.shape[:2]
    dst_h = int(crop_h*rich_crop_scale)
    dst_w = int(crop_w*rich_crop_scale)
    crop_img_scale = np.zeros((dst_h,dst_w,3),dtype=crop_img.dtype)
    crop_img_scale[:,:,:] = 127
    crop_seg_scale = np.zeros((dst_h,dst_w),dtype=crop_seg.dtype)
    dh = np.random.randint(0,(dst_h - crop_h + 1))
    dw = np.random.randint(0,(dst_w - crop_w + 1))
    crop_img_scale[dh:(dh+crop_h),dw:(dw+crop_w)] = crop_img_a
    crop_seg_scale[dh:(dh+crop_h),dw:(dw+crop_w)] = crop_seg_a
    return crop_img_scale,crop_seg_scale

def compose(img,grt):
    #首先resize 到src_shape
    img = cv2.resize(img, cfg["src_shape"], interpolation=cv2.INTER_LINEAR)
    grt = cv2.resize(grt, cfg["src_shape"], interpolation=cv2.INTER_NEAREST)
    #亮度，对比度，饱和度调整
    saturation_ratio = np.random.uniform(-0.5, 0.5)
    brightness_ratio = np.random.uniform(-0.5, 0.5)
    contrast_ratio   = np.random.uniform(-0.5, 0.5)
    order = [0,1,2,3,4,5]
    np.random.shuffle(order)
    for i in range(3):
        if order[i] == 0:
            img = saturation_jitter(img,saturation_ratio)
        if order[i] == 1:
            img = brightness_jitter(img,brightness_ratio)
        if order[i] == 2:
            img = contrast_jitter(img,contrast_ratio)

    #旋转
    rotation_ratio = np.random.uniform(-15,15)
    img,grt = random_rotation(img,grt,rotation_ratio)
    #aspect
    rich_crop_aspect_ratio = np.random.uniform(1.0,1.3)
    rich_crop_scale = np.random.uniform(1.0,1.4)
    img,grt = random_aspect(img,grt,rich_crop_scale,rich_crop_aspect_ratio)

    #mirror
    #if np.random.randint(0, 2) == 1:
    #    img = img[:, ::-1, :]
    #    grt = grt[:, ::-1]
    img = cv2.resize(img, cfg["dst_shape"], interpolation=cv2.INTER_LINEAR)
    grt = cv2.resize(grt, cfg["dst_shape"], interpolation=cv2.INTER_NEAREST)
    return img,grt

def crop_eval(img,grt):

    img = cv2.resize(img, cfg["src_shape"], interpolation=cv2.INTER_LINEAR)
    if grt is not None:
        grt = cv2.resize(grt, cfg["src_shape"], interpolation=cv2.INTER_NEAREST)
    pad_width = cfg["dst_shape"][0] - cfg["src_shape"][0]
    pad_height= cfg["dst_shape"][1] - cfg["src_shape"][1]
    p_w,p_h   = cfg["zero_pads"]

    if (pad_height > 0 or pad_width > 0):
        crop_img = cv2.copyMakeBorder(
            img,
            p_h,
            pad_height - p_h,
            p_w,
            pad_width - p_w,
            cv2.BORDER_CONSTANT,
            value=[127, 127, 127])
        if grt is not None:
            crop_grt = cv2.copyMakeBorder(
                grt,
                p_h,
                pad_height - p_h,
                p_w,
                pad_width - p_w,
                cv2.BORDER_CONSTANT,
                #value=cfg.DATASET.IGNORE_INDEX)
                value=0)
        else:
            crop_grt = None
        return crop_img,crop_grt
    elif pad_height == 0 or pad_width == 0:
        print(img.shape)
        print(grt.shape)
        return img,grt
    else:
        raise ValueError("Padding Error!")

if __name__ == "__main__":
    img = cv2.imread("/data/ai_lane/trainval_pic/10021517.jpg",-1)
    grt = cv2.imread("/data/ai_lane/trainval_tag/10021517.png",-1)
    ratios,hs,ws = [],[],[]
    #查看旋转后放大比例
    #Iter: 10000 Current ratio: 1.033008 average ratio: 1.296265
    #放大倍数1.1482
    #Iter:10000 Current ratio:1.235586 average ratio:1.296355 hs:1.218148 ws:1.060928
    for i in range(10000):
        img, grt = compose(img, grt)
        box_area = img.shape[0]*img.shape[1]
        area_ratio = box_area/(640*360)
        ratios.append(area_ratio)
        hs.append(img.shape[0]/360)
        ws.append(img.shape[1]/640)
        avg_ratio = np.sum(ratios)/len(ratios)
        avg_hs = np.sum(hs)/len(hs)
        avg_ws = np.sum(ws)/len(ws)
        print("Iter:%d Current ratio:%f average ratio:%f hs:%f ws:%f"%((i+1),area_ratio,avg_ratio,avg_hs,avg_ws))



