# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import cv2
#import matplotlib.pyplot as plt
import pydicom
import os,glob
import gc
def sobel_demo(image):
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)   #对x求一阶导
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)   #对y求一阶导
    gradx = cv2.convertScaleAbs(grad_x)  #用convertScaleAbs()函数将其转回原来的uint8形式
    grady = cv2.convertScaleAbs(grad_y)
#    cv2.imshow("gradient_x", gradx)  #x方向上的梯度
#    cv2.imshow("gradient_y", grady)  #y方向上的梯度
    gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0) #图片融合
    
    return gradxy

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

base_path = './data/Mindray'
i1 = os.listdir(base_path)
for i in i1:
    second_path = base_path + '/' + i 
    rpath = glob.glob(os.path.join(second_path,"*.png"))+glob.glob(os.path.join(second_path,"*.jpg"))+glob.glob(os.path.join(second_path ,"*.bmp"))
    
    for infile in rpath:
        print(infile)
        img = cv_imread(infile)
        gradient = sobel_demo(img.copy())
        #cv2.imshow("gradient", gradient)
        save_path = second_path + '/' + 'grad'
        if os.path.isdir(save_path)==True:
            pass
        else:
            os.makedirs(save_path)
        dt=(os.path.split(infile)[-1])
        
        cv2.imencode('.jpg',gradient)[1].tofile(save_path+'/'+dt)
        #cv.imwrite(name, img_cv)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img1 = img.reshape((img.shape[0]*img.shape[1],1))
        img1 = np.float32(img1)
        
        #define criteria = (type,max_iter,epsilon)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
        
        #set flags: hou to choose the initial center
        #---cv2.KMEANS_PP_CENTERS ; cv2.KMEANS_RANDOM_CENTERS
        flags = cv2.KMEANS_RANDOM_CENTERS
        # apply kmenas
        compactness,labels,centers = cv2.kmeans(img1,2,None,criteria,50,cv2.KMEANS_PP_CENTERS)
        if sum(labels==0)<sum(labels==1):
            labels[labels==0] = 2
            labels[labels==1] = 0
            labels[labels==2] = 1
        
        img2 = labels.reshape((img.shape[0],img.shape[1]))
        img2 = img2.astype(np.uint8)
        del compactness,labels,centers
        save_path1 = second_path + '/' + 'kmean'
        if os.path.isdir(save_path1)==True:
            pass
        else:
            os.makedirs(save_path1)
        cv2.imencode('.jpg',img2*255)[1].tofile(save_path1+'/'+dt)
