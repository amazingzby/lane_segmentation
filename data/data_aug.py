import cv2
import numpy as np
from config import lane_cfg as cfg

pad_value = [127.5,127.5,127.5]

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
    ratio = cfg["src_shape"][0]/cfg["src_shape"][1]
    nw = max(nw,int(nh*ratio))
    nh = max(nh,int(nw/ratio))

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
            borderValue=pad_value)
    crop_seg = cv2.warpAffine(
            crop_seg,
            r,
            dsize=dsize,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
    return crop_img, crop_seg

def random_aspect(crop_img,crop_seg,aspect_ratio):
    rand_num = np.random.uniform(0,1)
    height,width = crop_img.shape[:2]
    if rand_num < 0.5:
        n_h = int(height * np.random.uniform(aspect_ratio,1))
        n_w = int(width * np.random.uniform(aspect_ratio,1))
        p_h = np.random.randint(0,height- n_h + 1)
        p_w = np.random.randint(0,width - n_w + 1)
        crop_img = crop_img[p_h:(p_h+n_h),p_w:(p_w+n_w)]
        crop_seg = crop_seg[p_h:(p_h+n_h),p_w:(p_w+n_w)]
    else:
        aspect_ratio = 1/aspect_ratio
        n_h = int(height * np.random.uniform(1,aspect_ratio))
        n_w = int(width * np.random.uniform(1,aspect_ratio))
        p_h_1 = np.random.randint(0,n_h - height  + 1)
        p_w_1 = np.random.randint(0,n_w - width   + 1)
        p_h_2 = n_h - height - p_h_1
        p_w_2 = n_w - width  - p_w_1
        crop_img = cv2.copyMakeBorder(
            crop_img,
            p_h_1,
            p_h_2,
            p_w_1,
            p_w_2,
            cv2.BORDER_CONSTANT,
            value=pad_value)
        if crop_seg is not None:
            crop_seg = cv2.copyMakeBorder(
                crop_seg,
                p_h_1,
                p_h_2,
                p_w_1,
                p_w_2,
                cv2.BORDER_CONSTANT,
                value=0)
    return crop_img,crop_seg

def random_area(crop_img,crop_seg):
    height,width = crop_img.shape[:2]
    box_area1 = np.random.uniform(1,2.78)
    box_area2 = np.random.uniform(0.36,1)
    rand_choose = np.random.uniform(0,1)
    if rand_choose < 0.5:
        box_area = box_area1
    else:
        box_area = box_area2
    img_area = height * width
    dst_area = box_area * cfg["src_shape"][0] * cfg["src_shape"][1]
    scale = np.sqrt(dst_area / img_area)
    if scale < 1:
        n_h = int(height*scale)
        n_w = int(width*scale)
        p_h = np.random.randint(0,height- n_h + 1)
        p_w = np.random.randint(0,width - n_w + 1)
        crop_img = crop_img[p_h:(p_h+n_h),p_w:(p_w+n_w)]
        crop_seg = crop_seg[p_h:(p_h+n_h),p_w:(p_w+n_w)]
    elif scale > 1:
        n_h = int(height*scale)
        n_w = int(width*scale)
        p_h_1 = np.random.randint(0,n_h - height  + 1)
        p_w_1 = np.random.randint(0,n_w - width   + 1)
        p_h_2 = n_h - height - p_h_1
        p_w_2 = n_w - width  - p_w_1
        crop_img = cv2.copyMakeBorder(
            crop_img,
            p_h_1,
            p_h_2,
            p_w_1,
            p_w_2,
            cv2.BORDER_CONSTANT,
            value=pad_value)
        crop_seg = cv2.copyMakeBorder(
                crop_seg,
                p_h_1,
                p_h_2,
                p_w_1,
                p_w_2,
                cv2.BORDER_CONSTANT,
                value=0)
    return crop_img,crop_seg


def compose(img,grt):
    #首先resize 到src_shape
    img = cv2.resize(img, cfg["src_shape"], interpolation=cv2.INTER_LINEAR)
    grt = cv2.resize(grt, cfg["src_shape"], interpolation=cv2.INTER_NEAREST)
    keep_ratio = np.random.uniform(0, 1)
    if keep_ratio <= 0.1:
        return img,grt
    #亮度，对比度，饱和度调整
    saturation_ratio = np.random.uniform(-0.5, 0.5)
    brightness_ratio = np.random.uniform(-0.5, 0.5)
    contrast_ratio   = np.random.uniform(-0.5, 0.5)
    order = [0,1,2,3]
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

    #aspect ratio scale 0.7 宽高比随机缩放
    aspect_ratio = 0.7
    img,grt = random_aspect(img,grt,aspect_ratio)

    #面积
    #area_ratio = 0.6
    img,grt = random_area(img,grt)

    #resize
    img = cv2.resize(img, cfg["dst_shape"], interpolation=cv2.INTER_LINEAR)
    grt = cv2.resize(grt, cfg["dst_shape"], interpolation=cv2.INTER_NEAREST)
    return img,grt

def crop_eval(img,grt):
    #resize
    img = cv2.resize(img, cfg["dst_shape"], interpolation=cv2.INTER_LINEAR)
    if grt is not None:
        grt = cv2.resize(grt, cfg["dst_shape"], interpolation=cv2.INTER_NEAREST)
    return img,grt

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
