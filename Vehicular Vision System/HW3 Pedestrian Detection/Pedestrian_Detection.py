#importing libraries
from os import listdir
from os.path import isfile, join
import math
import cv2
import numpy as np
import time

# Path to folders
mypath = 'HW3_img'
mypath_MOG2Detection = 'MOG2_Detection'
mypath_HOGSVM = 'HOG_SVM'
mypath_MOG2HOG = 'MOG2HOG'

# variables for reading input folder as images
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
images = np.empty(len(onlyfiles), dtype=object)

# variables for Background Substraction
fgbg = cv2.createBackgroundSubtractorMOG2(history=4000)
kernel = cv2. getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_dil = np.ones((20, 20), np.uint8)

# variables for HOG+SVM
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# variable to choose the mode of program
print(" This program made for Vehicular Vision Systems course in NCTU as Homework.\nThe goal of this Homework is to create pedestrian detection program.\nThis program have 3 modes of execution:\n1)Background substraction\n2)HOG+SVM\n3)Background Substraction + HOG + SVM\nYou can select the mode by entering number 1(MOG2), 2(HOG+SVM) or 3(MOG2+HOG+SVM)\nIf you want more accurate result, enter 1(Background Substraction) or 2(HOG)\nOutput files will be in folders nearby .py program file.\nIf you want to change input files, put them into HW3_img folder\n")
mode = int(input("Please choose the mode by entering number 1(MOG2), 2(HOG) or 3(MOG2+HOG): "))


# Function for non maxima suppression for removal overlapping bounding boxes in HOG+SVM Method
def non_max_suppression_fast(boxes, overlap_thresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype("int")


# Pedestrian Detection based on MOG2 Background Substraction
if mode == 1:
    print("Please wait a little when Background Substraction will be prepared")
    for n in range(0, 500):
        images[n] = cv2.imread(join(mypath, onlyfiles[n]))
        mog_image = fgbg.apply(images[n])
    for name in onlyfiles:
        start = time.time()
        image = cv2.imread(join(mypath, name))
        mog_image = fgbg.apply(image)
        mog_image = cv2.morphologyEx(mog_image, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(mog_image, kernel_dil, iterations=1)
        (_, contours, _) = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 3000:
                x, y, w, h = cv2.boundingRect(contour)
                img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi = image[y:y-10+h+5, x:x-8+w+10]
        cv2.imwrite(join(mypath_MOG2Detection, name), image)
        stop = time.time()
        duration = stop - start
        print("[Pedestrian Detection using Background Substraction] {}: , Time {}".format(
            name, duration))
        cv2.waitKey(0)

# Pedestrian Detection based on HOG+SVM method
if mode == 2:
    for name in onlyfiles:
        start = time.time()
        image = cv2.imread(join(mypath, name))
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        orig = image.copy()
        (rects, weights) = hog.detectMultiScale(image, winStride=(8, 8),
                                                padding=(32, 32), scale=1.01)
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression_fast(rects, overlap_thresh=0.5)
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.imwrite(join(mypath_HOGSVM, name), image)
        stop = time.time()
        duration = stop - start
        print("[HOG+SVM Pedestrian Detection] {}: {} amount of boxes, Time {}".format(
            name, len(pick), duration))
        cv2.waitKey(0)

# Pedestrian Detection based on HOG+SVM with Background Substraction
if mode == 3:
    print("Please wait a little when Background Substraction will be prepared")
    for n in range(0, 500):
        images[n] = cv2.imread(join(mypath, onlyfiles[n]))
        mog_image = fgbg.apply(images[n])
    for name in onlyfiles:
        start = time.time()
        image = cv2.imread(join(mypath, name))
        mog_image = fgbg.apply(image)
        mog_image = cv2.morphologyEx(mog_image, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(mog_image, kernel_dil, iterations=1)
        orig = image.copy()
        orig_mog = dilation.copy()
        (rects, weights) = hog.detectMultiScale(image, winStride=(16, 16),
                                                padding=(32, 32), scale=1.05)
        (rects_mog, weights_mog) = hog.detectMultiScale(dilation, winStride=(16, 16),
                                                        padding=(32, 32), scale=1.05)
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
        for (x, y, w, h) in rects_mog:
            cv2.rectangle(orig_mog, (x, y), (x + w, y + h), (0, 0, 255), 2)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        rects_mog = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects_mog])
        pick = non_max_suppression_fast(rects, overlap_thresh=0.5)
        pick_mog = non_max_suppression_fast(rects_mog, overlap_thresh=0.5)
        if len(pick_mog) < len(pick) or len(pick_mog) == len(pick):
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        elif len(pick_mog) > len(pick):
            for (xA, yA, xB, yB) in pick_mog:
                cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        if len(pick_mog) < len(pick) or len(pick_mog) == len(pick):
            cv2.imwrite(join(mypath_MOG2HOG, name), image)
        elif len(pick_mog) > len(pick):
            cv2.imwrite(join(mypath_MOG2HOG, name), image)
        stop = time.time()
        duration = stop - start
        print("[INFO] {}: {} boxes without MOG2, {} boxes with MOG2, Time {}".format(
            name, len(pick), len(pick_mog), duration))
        cv2.waitKey(0)
