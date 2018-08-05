"""
This is a program for background substraction made. This program made for
Vehicular Vision System course. If you want to change input files, just add files
to "input" folder in project folder. Output will be in MOG, MOG2 and GMG folders.
If you want to work with another folders. Please change the path in appropriate
variables. Also don't forget to change path to files for MSE and PSNR computation.
You can do it in Truth_img, Input_img_MOG, Input_img_MOG2, Input_img_GMG.
"""

# importing libraries
from os import listdir
from os.path import isfile, join
import math
import cv2
import numpy as np

# Path to folders
mypath = 'HW2_img/input'
mypath_mog = 'HW2_img/MOG'
mypath_mog2 = 'HW2_img/MOG2'
mypath_gmg = 'HW2_img/GMG'

# variables for reading input folder as images
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
images = np.empty(len(onlyfiles), dtype=object)

# variables for background subtraction models
fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG(history=800)
fgbg2 = cv2.createBackgroundSubtractorMOG2(history=2000)
fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG(decisionThreshold=0.75)

# Elliptical kernel for morphological opening
kernel_gmg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# loop for MOG function learning
for n in range(0, 200):
    images[n] = cv2.imread(join(mypath, onlyfiles[n]))
    mog_image = fgbg1.apply(images[n])

# loop for MOG background subtraction
for name in onlyfiles:
    images[n] = cv2.imread(join(mypath, name))
    mog_image = fgbg1.apply(images[n])
    res = mog_image
    cv2.imwrite(join(mypath_mog, name), res)
    print(name)
    cv2.waitKey(2000)

print("Gaussian mixture model ver. 1 function completed")

# loop for MOG2 function learning
for n in range(0, 200):
    images[n] = cv2.imread(join(mypath, onlyfiles[n]))
    mog_image = fgbg2.apply(images[n])

# loop for MOG2 background subtraction
for name in onlyfiles:
    images[n] = cv2.imread(join(mypath, name))
    mog_image = fgbg2.apply(images[n])
    cv2.imwrite(join(mypath_mog2, name), mog_image)
    print(name)
    cv2.waitKey(2000)

print("Gaussian mixture model ver. 2 function completed")

# loop for GMG function learning
for n in range(0, 200):
    images[n] = cv2.imread(join(mypath, onlyfiles[n]))
    gmg_image = fgbg3.apply(images[n])
    gmg_image = cv2.morphologyEx(gmg_image, cv2.MORPH_OPEN, kernel_gmg)

# loop for GMG background subtraction
for name in onlyfiles:
    images[n] = cv2.imread(join(mypath, name))
    gmg_image = fgbg3.apply(images[n])
    gmg_image = cv2.morphologyEx(gmg_image, cv2.MORPH_OPEN, kernel_gmg)
    cv2.imwrite(join(mypath_gmg, name), gmg_image)
    print(name)
    cv2.waitKey(2000)

print("Gaussian mixture graphical model function completed")


# Function for MSE computation
def mse(image1, image2):
    mse = 0
    rows, cols, ch = image1.shape
    for i in range(rows):
        for j in range(cols):
            mse = mse + (image1[i, j]-image2[i, j])*(image1[i, j]-image2[i, j])
    mse = int(mse[0])+int(mse[1])+int(mse[2])
    mse = float(mse)/3
    mse = mse/(rows*cols)
    return mse


# function for PSNR computation
def psnr(image1, image2):
    psnr = 10 * math.log((255 * 255) / mse(image1, image2), 10)
    return psnr


# Output of MSE and PSNR in the end of program
Truth_img = cv2.imread("HW2_img/groundtruth/gt000002.png")
Input_img_MOG = cv2.imread("HW2_img/MOG/in000002.jpg")
Input_img_MOG2 = cv2.imread("HW2_img/MOG2/in000002.jpg")
Input_img_GMG = cv2.imread("HW2_img/GMG/in000002.jpg")
print("MSE FOR MOG  =", mse(Truth_img, Input_img_MOG))
print("PSNR FOR MOG  =", psnr(Truth_img, Input_img_MOG))
print("MSE FOR MOG2  =", mse(Truth_img, Input_img_MOG2))
print("PSNR FOR MOG2  =", psnr(Truth_img, Input_img_MOG2))
print("MSE FOR GMG  =", mse(Truth_img, Input_img_GMG))
print("PSNR FOR GMG  =", psnr(Truth_img, Input_img_GMG))
