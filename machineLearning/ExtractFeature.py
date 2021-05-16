import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from scipy import signal
import scipy.ndimage as ndimage
import os
import csv
from glob import glob
import pandas as pd

def getRoi(img):
    g = cv.split(img)[1]

    g = cv.GaussianBlur(g,(15,15),0)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(15,15))
    g = ndimage.grey_opening(g, structure = kernel)
    (minVal,maxVal,minLoc,maxLoc) = cv.minMaxLoc(g)

    y0 = int(maxLoc[1])-180
    y1 = int(maxLoc[1])+180
    x0 = int(maxLoc[0])-180
    x1 = int(maxLoc[0])+180
    crop = img[y0:y1,x0:x1]

    return crop

def delVessel(image):
    blue,green,red = cv.split(image)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(26,26))
    ves = cv.morphologyEx(green, cv.MORPH_BLACKHAT, kernel)

    vessel2 = cv.bitwise_or(ves,green)
    vessel = cv.bitwise_or(ves,red)
    vessel = cv.medianBlur(red,7)
    plt.imshow(vessel2)

    return vessel

def GetDisc(image):
    M = 60    #filter size

    filter = signal.gaussian(M, std=7) #Gaussian Window
    filter=filter/sum(filter)
    STDf = filter.std()  #It'standard deviation

    image_pre = image-image.mean()-image.std()

    thr = (0.5 * M) - (2*STDf) - image_pre.std()

    r,c = image.shape
    Dd = np.zeros(shape=(r,c))

    for i in range(1,r):
        for j in range(1,c):
            if image_pre[i,j]>thr:
                Dd[i,j]=255
            else:
                Dd[i,j]=0
    
    Dd = cv.morphologyEx(Dd, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2)), iterations = 1)
    Dd = cv.morphologyEx(Dd, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7)), iterations = 1)
    Dd = cv.morphologyEx(Dd, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(1,21)), iterations = 1)
    Dd = cv.morphologyEx(Dd, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(21,1)), iterations = 1)
    Dd = cv.morphologyEx(Dd, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(23,23)), iterations = 1)	
    Dd = cv.morphologyEx(Dd, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(43,43)), iterations = 1)

    Dd = np.uint8(Dd)

    return Dd

def GetCup(image):
    blue, green, red = cv.split(image)
    green = cv.medianBlur(green,7)

    M = 60    #filter size

    filter = signal.gaussian(M, std=7) #Gaussian Window
    filter = filter/sum(filter)
    STDf = filter.std()  #It'standard deviation

    green_pre = green-green.mean()-green.std()

    thr = (0.5 * M) + (2 * STDf) + (green_pre.std()) + (green_pre.mean())
    r,c = green.shape
    Dc = np.zeros(green.shape[:2])

    for i in range(1,r):
        for j in range(1,c):
            if green_pre[i,j]>thr:
                Dc[i,j]=255
            else:
                Dc[i,j]=0
    
    Dc = cv.morphologyEx(Dc, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2)), iterations = 1)
    Dc = cv.morphologyEx(Dc, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7)), iterations = 1)
    Dc = cv.morphologyEx(Dc, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(1,21)), iterations = 1)
    Dc = cv.morphologyEx(Dc, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(21,1)), iterations = 1)
    Dc = cv.morphologyEx(Dc, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(33,33)), iterations = 1)	
    Dc = cv.morphologyEx(Dc, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(33,33)), iterations = 1)

    Dc = np.uint8(Dc)
    return Dc

def circlecanvas(img,contour):
    (x,y) , radius = cv.minEnclosingCircle(contour)
    center = (int(x),int(y))
    radius = int(radius)
    final = cv.circle(img,center,radius,(0,255,0),1)

    return final

def cdr(oc,od):
    if (od == 0 or oc == 0):
        area = 0
    else:
        area = oc/od
    return area

def distancecd(oc,od):
    if (od == 0 or oc == 0):
        distance = 0
    else:
        distance = od - oc
    return distance

def extract(img):
    #crop and resize
    #if size > 1024 use resize
    #width = int(img.shape[1] * 0.5)
    #height = int(img.shape[0] * 0.5)
    #size = (width,height)
    #resized = cv.resize(img,size)
    #crop = getRoi(resized)
    
    crop = getRoi(img)
    vessel = delVessel(crop)

    #get disc & cup
    disc = GetDisc(vessel)
    cup = GetCup(crop)
    canny_disc = cv.Canny(disc,175,125)
    canny_cup = cv.Canny(cup,175,125)

    #draw disc
    contoursDisc = cv.findContours(canny_disc, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
    if len(contoursDisc) != 0 :
        od = 0
        for i in range(len(contoursDisc)):
            if od < cv.contourArea(contoursDisc[i]):
                od = cv.contourArea(contoursDisc[i])
                try:
                    canvas = circlecanvas(crop,contoursDisc[i])
                except(IndexError,ValueError,TypeError,AttributeError,EOFError,InterruptedError):
                    pass
    else:
        od = 0

    #draw cup
    contoursCup = cv.findContours(canny_cup, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
    if len(contoursCup) != 0 :
        oc = cv.contourArea(contoursCup[0])
        try:
            canvas = circlecanvas(crop,contoursCup[0])
        except(IndexError,ValueError,TypeError,AttributeError,EOFError,InterruptedError):
            pass
    else:
        oc = 0

    area = cdr(oc,od)

    distance = distancecd(oc,od)


    return area,distance


# make dataset
listCDr = []
listCDd = []
eye = []

#srcfolder
imgnames = sorted(glob("glaucoma/*.jpg"))
i = 0
while (i<len(imgnames)) :
    img = cv.imread(imgnames[i])
    x , y = extract(img)
    listCDr.append(x)
    listCDd.append(y)
    eye.append(0)
    print(i)
    i+=1

imgnames = sorted(glob("normal/*.jpg"))
i = 0
while (i<len(imgnames)) :
    img = cv.imread(imgnames[i])
    x , y = extract(img)
    listCDr.append(x)
    listCDd.append(y)
    eye.append(1)
    print(i)
    i+=1

imgnames = sorted(glob("other/*.jpg"))
i = 0
while (i<len(imgnames)) :
    img = cv.imread(imgnames[i])
    x , y = extract(img)
    listCDr.append(x)
    listCDd.append(y)
    eye.append(2)
    print(i)
    i+=1

#save dataset
df = pd.DataFrame({
    "CDr":listCDr,
    "CDd":listCDd,
    "eye":eye})

#saveset to csv
df.to_csv(r'.\eyedata.csv',index=False)


