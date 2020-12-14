import cv2
#[290, 39, 259, 40, 26, 20, 34, 36, 35, 20, 4, 0, 9, 22, 8, 2, 3, 1, 18]
#iou 0.2
#[569, 82, 543, 77, 36, 30, 57, 65, 61, 35, 8, 0, 22, 41, 11, 3, 4, 3, 44]
if __name__ == "__main__":
    f = open("train_ious.txt")
    lines = f.readlines()
    f.close()
    tag_dir = "/data/ai_lane/trainval_tag/"
    class_list = [0]*19
    count = 0
    for line in lines:
        count += 1
        print("%d image has processed!"%count)
        img_path = tag_dir + line.split()[0]
        iou = float(line.split()[1])
        if iou > 0.2:
            break
        img = cv2.imread(img_path,-1)
        class_ids = set(img.reshape(-1))
        for class_id in class_ids:
            if class_id != 0:
                class_list[class_id-1] += 1
    print(class_list)