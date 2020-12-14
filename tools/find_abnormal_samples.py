import os
import cv2
import numpy as np
from scipy.sparse import csr_matrix

pred_dir = "/data_bk/ai_lane/res_hrnet_medium/"
label_dir= "/data/ai_lane/trainval_tag_crop/"

def get_second_elem(elem):
    return elem[1]

def calculate(pred, label, num_classes=20):
    pred = pred.reshape(-1).astype(np.int64)
    label = label.reshape(-1).astype(np.int64)
    one = np.ones_like(pred).astype(np.int64)
    spm = csr_matrix((one, (label, pred)), shape=(num_classes, num_classes))
    spm = spm.todense()
    return spm

def mean_iou(confusion_matrix,num_classes=20):
    iou_list = []
    avg_iou = 0
    vji = np.zeros(num_classes, dtype=int)
    vij = np.zeros(num_classes, dtype=int)
    for j in range(num_classes):
        v_j = 0
        for i in range(num_classes):
            v_j += confusion_matrix[j,i]
        vji[j] = v_j

    for i in range(num_classes):
        v_i = 0
        for j in range(num_classes):
            v_i += confusion_matrix[j,i]
        vij[i] = v_i

    total = 0
    true = 0
    for c in range(1,num_classes):
        total += vji[c] + vij[c] - confusion_matrix[c,c]
        true += confusion_matrix[c,c]
    return float(true)/float(total)
    # for c in range(num_classes):
    #     total = vji[c] + vij[c] - confusion_matrix[c,c]
    #     if total == 0:
    #         iou = 0
    #     else:
    #         iou = float(confusion_matrix[c,c]) / total
    #     avg_iou += iou
    #     iou_list.append(iou)
    # avg_iou = float(avg_iou) / float(num_classes)
    # return np.array(iou_list), avg_iou

if __name__ == "__main__":
    img_files = os.listdir(pred_dir)
    iou_scores = []
    print("Total samples:%d\n"%len(img_files))
    count = 1
    for path in img_files:
        pred_path = os.path.join(pred_dir,path)
        label_path= os.path.join(label_dir,path)
        img1 = cv2.imread(pred_path)
        img2 = cv2.imread(label_path)
        confusion_matrix = calculate(img1,img2)
        avg_iou = mean_iou(confusion_matrix)
        iou_scores.append([path,avg_iou])
        print("Processing image %d:%s,iou:%f"%(count,path,avg_iou))
        count += 1
    iou_scores.sort(key=get_second_elem)
    f = open("analysisi_class_ids/train_ious.txt", "w")
    for sample in iou_scores:
        f.write(sample[0]+" "+str(sample[1])+"\n")
    f.close()
