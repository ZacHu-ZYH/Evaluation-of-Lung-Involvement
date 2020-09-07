# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:21:32 2020

@author: Administrator
"""

import pickle

f = open(r'D:\code\SENet-Tensorflow-master\cifar-10-batches-py-1\data_batch_1','rb') #以二进制读模式打开

d = pickle.load(f,encoding='bytes')


from PIL import Image

import numpy as np

import pickle,glob,os
from sklearn.model_selection import train_test_split

def cifar(path,classes):
    

    arr = [[]]
    
    #number of pictures
    
    n = 1
    rpath = glob.glob(path + "*.png")+glob.glob(path + "*.jpg")+glob.glob(path + "*.bmp")
    for infile in rpath:
    
        file,ext = os.path.splitext(infile)
        
        Img = Image.open(infile)
        
        print(Img.mode,file) 
    
        if Img.mode != 'RGB':
        
            Img = Img.convert('RGB')
            
            width = Img.size[0]
            
            height = Img.size[1]
        
            print('{} imagesize is:{} X {}'.format(n,width,height))
            
            n += 1
        
        Img = Img.resize([32,32],Image.ANTIALIAS)

        
        r,g,b = Img.split()    
        r_array = np.array(r).reshape([1024])
        g_array = np.array(g).reshape([1024])
        
        b_array = np.array(b).reshape([1024])
            
        merge_array = np.concatenate((r_array,g_array,b_array))
        merge_array = np.expand_dims(merge_array,0)
        
        if arr == [[]]:
        
            arr = merge_array
        arr = np.concatenate((arr,merge_array),axis=0)
        #arr = np.random.shuffle(arr)
    if classes==0:
        labelset = np.zeros((arr.shape[0],))
    elif classes==1:
        labelset = np.ones((arr.shape[0],)) 
    elif classes==2:
        labelset = np.ones((arr.shape[0],))*2
    elif classes==3:
        labelset = np.ones((arr.shape[0],))*3
    elif classes==4:
        labelset = np.ones((arr.shape[0],))*4
    elif classes==5:
        labelset = np.ones((arr.shape[0],))*5
    

    labelset = np.reshape(labelset,[arr.shape[0],])

    arr = np.concatenate((arr,arr,arr),axis=0)
    labelset = np.concatenate((labelset,labelset,labelset),axis=0)
    labelset= labelset.astype(np.uint8)
    return labelset,arr

path_A = r'E:\lung_data\arranged\章超\A线/'
path_A_B = r'E:\lung_data\arranged\章超\A线+B线/'
path_duofa_B = r'E:\lung_data\arranged\章超\多发B线/'
path_duofaronghe_B = r'E:\lung_data\arranged\章超\多发B线+融合B线/'
path_shibian = r'E:\lung_data\arranged\章超\肺实变\原图/'
path_ronghe_B = r'E:\lung_data\arranged\章超\融合B线/'

path_A_m = r'E:\lung_data\arranged\迈瑞\A线/'
path_A_B_m = r'E:\lung_data\arranged\迈瑞\A线+B线/'
path_duofa_B_m = r'E:\lung_data\arranged\迈瑞\多发B线/'
path_duofaronghe_B_m = r'E:\lung_data\arranged\迈瑞\多发B线+融合B线/'
path_shibian_m = r'E:\lung_data\arranged\迈瑞\肺实变\原图/'
path_ronghe_B_m = r'E:\lung_data\arranged\迈瑞\融合B线/'

path_duofa_B_f = r'E:\lung_data\arranged\飞利浦\多发B线/'
path_duofaronghe_B_f = r'E:\lung_data\arranged\飞利浦\多发+融合B线/'
path_shibian_f = r'E:\lung_data\arranged\飞利浦\肺实变/'
path_ronghe_B_f = r'E:\lung_data\arranged\飞利浦\融合B线/'
label_A,arr_A =cifar(path_A,0)
label_A_B,arr_A_B =cifar(path_A_B,1)
label_duofa_B,arr_A_duofa_B =cifar(path_duofa_B,2)
label_duofaronghe_B,arr_A_duofaronghe_B =cifar(path_duofaronghe_B,3)
label_A_shibian,arr_A_shibian =cifar(path_shibian,4)
label_A_ronghe_B,arr_A_ronghe_B =cifar(path_ronghe_B,5)

label_A_m,arr_A_m =cifar(path_A_m,0)
label_A_B_m,arr_A_B_m =cifar(path_A_B_m,1)
label_duofa_B_m,arr_A_duofa_B_m =cifar(path_duofa_B_m,2)
label_duofaronghe_B_m,arr_A_duofaronghe_B_m =cifar(path_duofaronghe_B_m,3)
label_A_shibian_m,arr_A_shibian_m =cifar(path_shibian_m,4)
label_A_ronghe_B_m,arr_A_ronghe_B_m =cifar(path_ronghe_B_m,5)

label_duofa_B_f,arr_A_duofa_B_f =cifar(path_duofa_B_f,2)
label_duofaronghe_B_f,arr_A_duofaronghe_B_f =cifar(path_duofaronghe_B_f,3)
label_A_shibian_f,arr_A_shibian_f =cifar(path_shibian_f,4)
label_A_ronghe_B_f,arr_A_ronghe_B_f =cifar(path_ronghe_B_f,5)

final_arr = np.concatenate((arr_A,arr_A_B,arr_A_duofa_B,arr_A_duofaronghe_B,arr_A_shibian,arr_A_ronghe_B,arr_A_m,arr_A_B_m,arr_A_duofa_B_m,arr_A_duofaronghe_B_m,arr_A_shibian_m,arr_A_ronghe_B_m,arr_A_duofa_B_f,arr_A_duofaronghe_B_f,arr_A_shibian_f,arr_A_ronghe_B_f),axis=0)
final_label = np.concatenate((label_A,label_A_B,label_duofa_B,label_duofaronghe_B,label_A_shibian,label_A_ronghe_B,label_A_m,label_A_B_m,label_duofa_B_m,label_duofaronghe_B_m,label_A_shibian_m,label_A_ronghe_B_m,label_duofa_B_f,label_duofaronghe_B_f,label_A_shibian_f,label_A_ronghe_B_f),axis=0)
x_train, y_train, x_test, y_test = train_test_split(final_arr, final_label, test_size=0.1, random_state=10)

train_dic = {b'data':x_train,b'labels':x_test}
test_dic = {b'data':y_train,b'labels':y_test}
f = open('./cifar-10-batches-py-1/data_batch_mix_train_addfeilipu','wb')
pickle.dump(train_dic,f,protocol=2)


g = open('./cifar-10-batches-py-1/test_batch_mix_addfeilipu','wb')

pickle.dump(test_dic,g,protocol=2)
