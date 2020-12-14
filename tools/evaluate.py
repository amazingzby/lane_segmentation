import numpy as np
from glob import glob
import cv2
from scipy.sparse import csr_matrix

class ConfusionMatrix(object):
    """
        Confusion Matrix for segmentation evaluation
    """

    def __init__(self, num_classes=2, streaming=False):
        self.confusion_matrix = np.zeros([num_classes, num_classes],
                                         dtype='int64')
        self.num_classes = num_classes
        self.streaming = streaming

    def calculate(self, pred, label, ignore=None):
        # If not in streaming mode, clear matrix everytime when call `calculate`
        if not self.streaming:
            self.zero_matrix()

        #label = np.transpose(label, (0, 2, 3, 1))
        #ignore = np.transpose(ignore, (0, 2, 3, 1))
        #mask = np.array(ignore) == 1

        pred = pred.reshape(-1).astype(np.int64)
        label= label.reshape(-1).astype(np.int64)
        # label = np.asarray(label)
        # pred = np.asarray(pred)
        one = np.ones_like(pred).astype(np.int64)
        # Accumuate ([row=label, col=pred], 1) into sparse matrix
        #label 行 pred列
        spm = csr_matrix((one, (label, pred)),shape=(self.num_classes, self.num_classes))
        spm = spm.todense()
        self.confusion_matrix += spm
        #self.confusion_matrix = spm

    def zero_matrix(self):
        """ Clear confusion matrix """
        self.confusion_matrix = np.zeros([self.num_classes, self.num_classes],
                                         dtype='int64')

    def mean_iou(self):
        iou_list = []
        avg_iou = 0
        # TODO: use numpy sum axis api to simpliy
        vji = np.zeros(self.num_classes, dtype=int)
        vij = np.zeros(self.num_classes, dtype=int)
        for j in range(self.num_classes):
            v_j = 0
            for i in range(self.num_classes):
                v_j += self.confusion_matrix[j][i]
            vji[j] = v_j

        for i in range(self.num_classes):
            v_i = 0
            for j in range(self.num_classes):
                v_i += self.confusion_matrix[j][i]
            vij[i] = v_i

        for c in range(self.num_classes):
            total = vji[c] + vij[c] - self.confusion_matrix[c][c]
            if total == 0:
                iou = 0
            else:
                iou = float(self.confusion_matrix[c][c]) / total
            avg_iou += iou
            iou_list.append(iou)
        avg_iou = float(avg_iou) / float(self.num_classes)
        return np.array(iou_list), avg_iou

    def accuracy(self):
        total = self.confusion_matrix.sum()
        total_right = 0
        for c in range(self.num_classes):
            total_right += self.confusion_matrix[c][c]
        if total == 0:
            avg_acc = 0
        else:
            avg_acc = float(total_right) / total

        vij = np.zeros(self.num_classes, dtype=int)
        for i in range(self.num_classes):
            v_i = 0
            for j in range(self.num_classes):
                v_i += self.confusion_matrix[j][i]
            vij[i] = v_i

        acc_list = []
        for c in range(self.num_classes):
            if vij[c] == 0:
                acc = 0
            else:
                acc = self.confusion_matrix[c][c] / float(vij[c])
            acc_list.append(acc)
        return np.array(acc_list), avg_acc

    def kappa(self):
        vji = np.zeros(self.num_classes)
        vij = np.zeros(self.num_classes)
        for j in range(self.num_classes):
            v_j = 0
            for i in range(self.num_classes):
                v_j += self.confusion_matrix[j][i]
            vji[j] = v_j

        for i in range(self.num_classes):
            v_i = 0
            for j in range(self.num_classes):
                v_i += self.confusion_matrix[j][i]
            vij[i] = v_i

        total = self.confusion_matrix.sum()

        # avoid spillovers
        # TODO: is it reasonable to hard code 10000.0?
        total = float(total) / 10000.0
        vji = vji / 10000.0
        vij = vij / 10000.0

        tp = 0
        tc = 0
        for c in range(self.num_classes):
            tp += vji[c] * vij[c]
            tc += self.confusion_matrix[c][c]

        tc = tc / 10000.0
        pe = tp / (total * total)
        po = tc / total

        kappa = (po - pe) / (1 - pe)
        return kappa

#评估验证集的mIoU
val_files = glob("/data_bk/ai_lane/res_deeplab_medium/*.png")
labelpath = "/data/ai_lane/train_tag_crop/"
conf_mat = ConfusionMatrix(20,streaming=True)
step = 0
for file in val_files:
    step += 1
    label_path = labelpath + file.split("/")[-1]
    img1 = cv2.imread(file,-1)
    img2 = cv2.imread(label_path,-1)
    conf_mat.calculate(img1, img2)
    _, iou = conf_mat.mean_iou()
    _, acc = conf_mat.accuracy()
    print(
        "[EVAL]step={} acc={:.4f} IoU={:.4f}"
        .format(step, acc, iou))
category_iou, avg_iou = conf_mat.mean_iou()
category_acc, avg_acc = conf_mat.accuracy()
print("[EVAL]Category IoU:", category_iou)
print("[EVAL]Category Acc:", category_acc)
print("[EVAL]Kappa:{:.4f}".format(conf_mat.kappa()))