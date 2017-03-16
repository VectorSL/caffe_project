# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 20:53:09 2016

@author: SL
"""

from collections import deque
import cv2
import cv2.cv as cv
import Image, ImageTk
import time
from tkMessageBox import *
from Tkinter import StringVar
import Tkinter as tk
from Tkinter import Entry as Entry
from Tkinter import Label as Label
import numpy as np
import matplotlib.pyplot as plt
import skimage
import random
caffe_root = 'C:/caffe-windows-master/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import sklearn.metrics.pairwise as pw
import linecache

def quit_(root):
    root.destroy()
def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3,
                                    minNeighbors=5, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
#    print rects
    return rects


def face(cam):
    (readsuccessful, img) = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    rects = detect(gray, cascade)
    vis = img.copy() 
    for x, y, w, h in rects:
        crop=vis[y:y+h,x:x+w]
    return crop

def Reset(root):
    '''
    重置信息，删除已有的ID及特征
    '''
    if askyesno('Warning', 'This command will clear all the members!'):
        showwarning('Yes', 'Reset successfully')
        f=file("../readfile/name_one_label.txt","w")
        f.close()
        f=file("../readfile/feature1.txt","w")
        f.close()
    else:
        showinfo('No', 'Reset has been cancelled')

def add_name():
    print name.get()
    f=file("../readfile/name_one_label.txt","a+")
    f.write('\n'+name.get())
    f.close()
    crop=face(cam)
    cv2.imwrite('../Addface/%s.jpg'%str(name.get()+str(random.randint(0,9))),crop)
    add_feature(crop)
    f=file("../readfile/temp.txt","a+")
    line=f.readlines()
    feature=file("../readfile/feature1.txt","a+")
    feature.writelines(line) 
    f.close()
    feature.close()

def Login():
    text='enter your name'
    label=Label(root,text=text).pack()

    Entry(root,textvariable=name).pack() #设置输入框对应的文本变量为var
    tk.Button(root,text="OK",command=add_name).pack()

def Refresh():
    global n
    n=1

def draw_rects(img, rects, color):
    for x, y, w, h in rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

def update_image(image_label, cam):
    (readsuccessful, f) = cam.read()
    gray_im = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    #gray_im=f
#    cascade_fn = 'haarcascade_frontalface_alt.xml'
#    cascade = cv2.CascadeClassifier(cascade_fn)
    rects = detect(gray_im, cascade)
    vis = gray_im.copy()

    #print evaluate(metric,vis)
    global n
    draw_rects(vis, rects, (0, 255, 0))
    if len(rects)>0:
        if n==8:
            var = tk.StringVar()
#            im2=face(cam)
            for x, y, w, h in rects:
                im2=f[y:y+h,x:x+w]
            #evaluate(metric,im2)
            var=evaluate(metric,im2)
            l.config(text=var)
            
            l.pack()
            #Label.forget()
        n=n+1
        
    a = Image.fromarray(vis)
    b = ImageTk.PhotoImage(image=a)
    image_label.configure(image=b)
    image_label._image_cache = b  # avoid garbage collection
    root.update()


def update_fps(fps_label):
    frame_times = fps_label._frame_times
    frame_times.rotate()
    frame_times[0] = time.time()
    sum_of_deltas = frame_times[0] - frame_times[-1]
    count_of_deltas = len(frame_times) - 1
    try:
        fps = int(float(count_of_deltas) / sum_of_deltas)
    except ZeroDivisionError:
        fps = 0
    fps_label.configure(text='FPS: {}'.format(fps))


def update_all(root, image_label, cam, fps_label):
    update_image(image_label, cam)
    update_fps(fps_label)
    root.after(20, func=lambda: update_all(root, image_label, cam, fps_label))



def read_feature(filelist):
    '''
    '''
    fid=open(filelist)
    lines=fid.readlines()
    test_num=len(lines)
    fid.close()
    X=np.empty((test_num,4096),float)
    i =0
    for line in lines:
        word=line.split(',')
        for j in range(0,4095):
            X[i,j]=word[j]
	#print caffe_root1+word[0]
	i=i+1
    return X

def read_labels(labelfile,k):
    '''
    读取标签列表文件
    '''
    linecache.clearcache()
    fin=open(labelfile,'r')
    lines=linecache.getline(labelfile,k)
    #labels=np.empty((len(lines),))
    fin.close()
    return lines

def evaluate(metric,im2):
    
    #设置为gpu格式
    caffe.set_mode_cpu()
    net = caffe.Classifier('../vgg_face_caffe1/VGG_FACE_deploy.prototxt', 
    '../vgg_face_caffe1/VGG_FACE.caffemodel',
    0)
    filelist_label='../readfile/name_one_label.txt'
    filelist='../readfile/feature1.txt'
    #im2=skimage.io.imread('F:/Data/GUI/face/chenfeng/chenfeng29.jpg',as_grey=False)
    image2=skimage.transform.resize(im2,(224, 224))*255
    Y=np.empty((1,3,224,224))
    Y[0,0,:,:]=image2[:,:,0]
    Y[0,1,:,:]=image2[:,:,1]
    Y[0,2,:,:]=image2[:,:,2]
    out = net.forward_all(data = Y )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    feature2 = np.float64(out['fc7'])
    feature2=np.reshape(feature2,(1,4096))
    X=read_feature(filelist)
    test_num=test_num=np.shape(X)[0]
    feature1=np.reshape(X,(test_num,4096))
    predict=tk.StringVar()
    mt=pw.pairwise_distances(feature1, feature2, metric=metric)
    distance=np.empty(test_num)
    for i in range(test_num):
          distance[i]=mt[i][0]
        # 距离需要归一化到0--1,与标签0-1匹配
    #print np.min(distance)
    if np.min(distance) <= 0.15 :
       #labels=read_labels(filelist_label)
       for i in range(test_num):
	     if distance[i]==np.min(distance):
		predict=read_labels(filelist_label,i+1)
		#print 'the distance is',np.min(distance)
		#print time.clock()
    else:
	predict='none'
    return predict


def add_feature(img):
    caffe.set_mode_cpu()
    net = caffe.Classifier('../vgg_face_caffe1/VGG_FACE_deploy.prototxt', 
    '../vgg_face_caffe1/VGG_FACE.caffemodel',
    0)
    image2=skimage.transform.resize(img,(224, 224))*255
    Y=np.empty((1,3,224,224))
    Y[0,0,:,:]=image2[:,:,0]
    Y[0,1,:,:]=image2[:,:,1]
    Y[0,2,:,:]=image2[:,:,2]
    out = net.forward_all(data = Y )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    feature2 = np.float64(out['fc7'])
    feature2=np.reshape(feature2,(1,4096))
    np.savetxt('../readfile/temp.txt', feature2, delimiter=',')
    
    
    
    
global cascade_fn
global cascade
 

if __name__ == '__main__':
    root = tk.Tk()
    # label for the video frame
    image_label = tk.Label(master=root)
    image_label.pack()
    
    cascade_fn = 'haarcascade_frontalface_alt.xml'
    cascade = cv2.CascadeClassifier(cascade_fn)   
    # camera
    cam = cv2.VideoCapture(0)
    metric='cosine'
    n=1
    # label for fps
    fps_label = tk.Label(master=root)
    fps_label._frame_times = deque([0]*5)  # arbitrary 5 frame average FPS
    fps_label.pack()
    l=tk.Label(root)
    select_button=tk.Button(master=root,text='Login',width=8,height=2,command=lambda:Login())
    select_button.pack(side='left',pady=20,padx=20)
    
    select_button=tk.Button(master=root,text='Refresh',width=8,height=2,command=lambda:Refresh())
    select_button.pack(side='left',pady=20,padx=20)
    
    select_button=tk.Button(master=root,text='Quit',width=8,height=2,command=lambda:quit_(root))
    # setup the update callback
    select_button.pack(side='right',pady=20,padx=20)

    select_button=tk.Button(master=root,text='Reset',width=8,height=2,command=lambda:Reset(root))
    select_button.pack(side='right',pady=20,padx=20)

    name=tk.StringVar()
    root.after(0, func=lambda: update_all(root, image_label, cam, fps_label))
    root.mainloop()