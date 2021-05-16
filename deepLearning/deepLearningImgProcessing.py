import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2 as cv
import scipy.ndimage as ndimage
import os
import csv
import random
import math
import time
import imutils
from os import listdir

# rotate img
# change other to glaucoma or normal to preprocess  glaucoma or normal
def rotate(folder,deg):
    count = 1792
    for file in os.listdir("imgprocessing/train/"+folder+"/"):
        img = cv.imread(os.path.join("imgprocessing/train/"+folder+"/"+file))
        #print(img)
        rows,cols,ch = img.shape
    
        M = cv.getRotationMatrix2D((cols/2,rows/2),deg,1)
    
        dst = cv.warpAffine(img,M,(cols,rows))
        filename = "otherRotate"+str(count)
        print(filename+"save completed")
        count = count+1
        cv.imwrite(r"C:\Users\User\DeepLearning\dataset\train\other\{}.jpg".format(filename), dst)


# rotate 90 degree
# change other to glaucoma or normal to preprocess  glaucoma or normal
def resize(folder):
    count = 0
    for file in os.listdir("dataset/train/"+folder+"/"):
        img = cv.imread(os.path.join("dataset/train/"+folder+"/"+file))
        resize = cv.resize(img, (224,224))
        filename = "normalResize"+str(count)
        print(filename+"save completed")
        count = count+1
        cv.imwrite(r"C:\Users\User\DeepLearning\imgprocessing\train\other\{}.jpg".format(filename), resize)


# resize img 255
# change other to glaucoma or normal to preprocess  glaucoma or normal
def noise(folder,prob):
    count = 0
    for file in os.listdir("imgprocessing/train/"+folder+"/"):
        img = cv.imread(os.path.join("imgprocessing/train/"+folder+"/"+file))
        output = np.zeros(img.shape,np.uint8)
        thres = 1 - prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = img[i][j]
        filename = "otherNoise"+str(count)
        print(filename+"save completed")
        count = count+1
        cv.imwrite(r"C:\Users\User\DeepLearning\dataset\train\other\{}.jpg".format(filename), output)


# gray scale
# change other to glaucoma or normal to preprocess  glaucoma or normal
def clahe(folder):
    count = 0
    for file in os.listdir("imgprocessing/train/"+folder+"/"):
        img = cv.imread(os.path.join("imgprocessing/train/"+folder+"/"+file))
        image_bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit = 3)
        final_img = clahe.apply(img)
        filename = "otherClaheC"+str(count)
        print(filename+" save completed")
        count = count+1
        cv.imwrite(r"C:\Users\User\DeepLearning\dataset\train\other\{}.jpg".format(filename), final_img)


# zoom img
# change other to glaucoma or normal to preprocess  glaucoma or normal
def crop(folder):
    count = 0
    for file in os.listdir("imgprocessing/train/"+folder+"/"):
        img = cv.imread(os.path.join("imgprocessing/train/"+folder+"/"+file))
        
        height = int(img.shape[0])
        width = int(img.shape[1])

        y=height/4
        x=width/4
        h=height/2
        w=width/2
        a = int(y+h)
        b = int(x+w)
        crop = img[int(y):a, int(x):b]
        
        filename = "otherCrop"+str(count)
        count = count+1
        cv.imwrite(r"C:\Users\User\DeepLearning\dataset\train\other\{}.jpg".format(filename), crop)
        print(filename+" save completed")


# darken img
# change other to glaucoma or normal to preprocess  glaucoma or normal
def darken(folder):
    count = 0
    for file in os.listdir("dataset/train/"+folder+"/"):
        img = cv.imread(os.path.join("dataset/train/"+folder+"/"+file))
        
        hsv = cv.cvtColor(img,cv.COLOR_RGB2HSV)
        hsv[...,2] = hsv[...,2]*0.6
        darken = cv.cvtColor(hsv,cv.COLOR_HSV2RGB)
        
        filename = "normalDarken"+str(count)
        count = count+1
        cv.imwrite(r"C:\Users\User\DeepLearning\dataset\train\other\{}.jpg".format(filename), darken)
        print(filename+" save completed")




