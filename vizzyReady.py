import cv2
import sys
import numpy as np
from joblib import load
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
from openpose import pyopenpose as op

#


if sys.argv[0][-4]==".":
    img = cv2.read(sys.argv[0])
else:
    img = sys.argv[0]
     
h, w = img.shape[:2] 
h = int(h) 
w= int(w)

if int(h/w) != int(16/9):
    blank_image = np.zeros(shape=[1280, 720, 3], dtype=np.uint8)
    img = cv2.addWeighted(img,1,blank_image,0,0)

def master(imagem):
     
    params = dict()
    params["model_folder"] = "/home/sims/repositories/openpose/models/"
      
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
    frame = cv2.resize(imagem,(1280 ,720))    
    datum = op.Datum()
    imageToProcess = frame
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum]) 
    ok = True 
    
    try:
        keypoints = (datum.poseKeypoints[0])
    except(IndexError):
        ok = False
        print('No people found')
    
    if ok:
        falhado = 0
        
        fim = [keypoints[0][0],keypoints[0][1],\
               keypoints[17][0],keypoints[17][1],\
               keypoints[18][0],keypoints[18][1],\
               keypoints[15][0],keypoints[15][1],\
               keypoints[16][0],keypoints[16][1]]
        
        fim = [-1 if x==0 else x for x in fim]
        for x in fim:
            if x == -1:
                falhado += 1

        if (sys.argv[1]) == "y":
            cv2.imshow('image',frame)
       
        cv2.waitKey(50)

        if falhado < 3:
            clf2 = load('classificador1.joblib')
            focus = (clf2.predict(np.array([fim])))
    return focus
  
def look(n):
    x,y=0
    if n in [0,2,8,10]:
        x = int((w*1)/8)
        
    elif n in [1,3,9,11]:
        x = int((w*3)/8)
        
    elif n in [4,6,12,14]:
        x = int((w*5)/8)
        
    elif n in [5,7,13,15]:
        x = int((w*7)/8)
        
    if n in [0,1,4,5]:
        y = int((h*1)/8)
        
    elif n in [2,3,6,7]:
        y = int((h*3)/8)
        
    elif n in [8,9,12,13]:
        y = int((h*5)/8)
        
    elif n in [10,11,14,15]:
        y = int((h*7)/8)
        
    z = 22
                
    return x,y,z

look(master(img))


    






