#Importing packages:
import cv2 #image manipulation
import sys #manipulating function arguments
import numpy as np #eventual mathemathic operations
from pydarknet import Detector, Image #openpose
import os #operating system interactions
dir_path = os.path.dirname(os.path.realpath(__file__))
from openpose import pyopenpose as op #open pose

# coords2Quadrante calculates to which quadrant (out of 16) a tuple of coordinates refers
# the function's parameters include the coordinates - a tuple - and the image object
# It outputs the quadrant and the image with the calculated quadrant overlined
def coords2Quadrante(coords,img):
    overlay = img.copy() # safe copy of the image to apply alpha
    h, w = img.shape[:2] # gets the image shape (height, width) 
    h = int(h) # converts parameters to integers 
    w= int(w)
    
    # next, it compares the x and the y to a reference from the original image (x/2,x/4 or y/2,y/4)
    # note that the image was devided in 16 rectangles: firstly divided in four and each of these, devided
    # in four again
    
    #____________________________# 0
    # 0  \  1   \   4   \   5    #
    #-----------\----------------# h/4
    # 2  \  3   \   6   \   7    #
    #___________\________________# h/2
    # 8  \  9   \   12  \   13   #
    # ----------\----------------# (3h)/4
    # 10 \  11  \   13  \   15   #
    # __________\________________#  h
    # 0  w/4    w/2    3w/4    w


    if coords[0]<int(w/2):
        if coords[1]<int(h/2):
            if coords[0]<int(w/4):
                if coords[1]<int(h/4):
                    quadrante=0
                    cv2.rectangle(overlay, (0,0), (int(w/4),int(h/4)), (229,88,191), thickness=-1) 
                else:
                    quadrante=2
                    cv2.rectangle(overlay, (0,int(h/4)), (int(w/4),int(h/2)), (229,88,191), thickness=-1)
            else:
                 if coords[1]<int(h/4):
                    quadrante=1
                    cv2.rectangle(overlay, (int(w/4),0), (int(w/2),int(h/4)), (229,88,191), thickness=-1)
                 else:
                    quadrante=3               
                    cv2.rectangle(overlay, (int(w/4),int(h/4)), (int(w/2),int(h/2)), (229,88,191), thickness=-1)
        else:
            if coords[0]<int(w/4): 
                if coords[1]<int((3*h)/4):
                    quadrante=8
                    cv2.rectangle(overlay, (0,int(h/2)), (int(w/4),int((3*h)/4)), (229,88,191), thickness=-1)
                    
                else:
                    quadrante=10
                    cv2.rectangle(overlay, (0,int((3*h)/4)), (int(w/4),int(h)), (229,88,191), thickness=-1)
            else:
                 if coords[1]<int((3*h)/4):
                    quadrante=9
                    cv2.rectangle(overlay, (int(w/4),int(h/2)), (int(w/2),int((3*h)/4)), (229,88,191), thickness=-1)
                 else:
                    quadrante=11               
                    cv2.rectangle(overlay, (int(w/4),int((3*h)/4)), (int(w/2),h), (229,88,191), thickness=-1)
    else:
        if coords[1]<int(h/2): 
            if coords[0]<int((3*w)/4):
                if coords[1]<int(h/4):
                    quadrante=4
                    cv2.rectangle(overlay, (int(w/2),0), (int((3*w)/4),int(h/4)), (229,88,191), thickness=-1)
                 
                else:
                    quadrante=6
                    cv2.rectangle(overlay, ((int(w/2),int(h/4))), (int((3*w)/4),int(h/2)), (229,88,191), thickness=-1)
                   
            else:
                 if coords[1]<int(h/4):
                    quadrante=5
                    cv2.rectangle(overlay, (int((3*w)/4),0), (int(w),int(h/4)), (229,88,191), thickness=-1)
                    
                 else:
                    quadrante=7          
                    cv2.rectangle(overlay, (int((3*w)/4),int(h/4)), (int(w),int(h/2)), (229,88,191), thickness=-1)
                    
        else:
            if coords[0]<int((3*w)/4): 
                if coords[1]<int((3*h)/4):
                    quadrante=12
                    cv2.rectangle(overlay, (int(w/2),int(h/2)), (int((3*w)/4),int((3*h)/4)), (229,88,191), thickness=-1)
                else:
                    quadrante=14
                    cv2.rectangle(overlay, (int(w/2),int((3*h)/4)), (int((3*w)/4),int(h)), (229,88,191), thickness=-1)
            else:
                 if coords[1]<int((3*h)/4):
                    quadrante=13
                    cv2.rectangle(overlay, (int((3*w)/4),int(h/2)), (int(w),int((3*h)/4)), (229,88,191), thickness=-1)
                 else:
                    quadrante=15               
                    cv2.rectangle(overlay, (int((3*w)/4),int((3*h)/4)), (int(w),int(h)), (229,88,191), thickness=-1)                    
            
    new = cv2.addWeighted(overlay, 0.4, img, 1 - 0.4, 0, img) #purple overlay 
    return quadrante,new #returns the quadrant and a new picture, with the underlined quadrant


# contextual information - classifying an object in long or short range interaction
def short_long(objeto):
    cena='cena'
    if objeto in ['tv','pottedplant','vase','cat']:
        cena='long'
    if objeto in ['chair','dining table','refrigerator','microwave','sink','bowl','apple','banana','laptop',\
                'keyboard','mouse','knife','fork','backpack']:
        cena='short'
    return cena

# master function, receives a video and the ground truth object
def master(video,objeto):
    cap = cv2.VideoCapture(video)
     
    
    #Initializing openPose, specifying directories
    params = dict()
    params["model_folder"] = "/home/sims/repositories/openpose/models/" 
      
    for i in range(0, len(video)):
        curr_item = video[i]
        if i != len(video)-1: next_item = video[i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item
     
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
     
    net = Detector(bytes("/home/sims/repositories/YOLO3-4-Py/cfg/yolov3.cfg", encoding="utf-8"), \
    bytes("/home/sims/repositories/YOLO3-4-Py/weights/yolov3.weights", encoding="utf-8"), 0, 
    bytes("/home/sims/repositories/YOLO3-4-Py/cfg/coco.data",encoding="utf-8"))  
     
    i = 0
    while(True):
        

        ret, frame = cap.read()
        #print(sys.argv[3])
        if ret:
            print(i) 
            falhado = 0
            

            frame = cv2.resize(frame,(1280 ,720))
            
            m = 0
            
            while m != 2 :
                
                if m == 1:
                    frame = cv2.flip(frame,1)
                
                if (sys.argv[3]) == "y":
                    cv2.imshow('image',frame)
                    
                cv2.waitKey(25)
                targetx = 0
                targety = 0
                 
                dark_frame = Image(frame)
                results = net.detect(dark_frame)
                conta=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
                
                
                for cat, score, bounds in results:
                    x, y, w, h = bounds
                    cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(0,0,255), thickness=2)
                    cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
                    if str(cat.decode("utf-8")) == objeto:
                        maxh = 0
                        if h > maxh:
                            targetx = x
                            targety = y
                            cv2.circle(frame, (int(x),int(y)), 3,(255,255,255), thickness=5)  
                    
                    
                    a=short_long(str(cat.decode("utf-8")))
                    
                    b=coords2Quadrante((x,y),frame)[0]
                    
                    
                    if a=='short':
                        conta[0][b]+=1
                    if a=='long':
                        conta[1][b]+=1
                        
                if (sys.argv[3]) == "y":
                    cv2.imshow('image',frame)
                cv2.waitKey(25)
                del dark_frame
                
                if targetx == 0:
                    falhado = 4
                #cv2.imwrite(sys.argv[1][:-4]+str(i)+"_YOLO.png",frame)
                datum = op.Datum()
                imageToProcess = frame
                #os.remove(sys.argv[1][:-4] +str(i)+"_YOLO.png")
                datum.cvInputData = imageToProcess
                opWrapper.emplaceAndPop([datum]) 
                
        
                try:
                    posi = (datum.poseKeypoints[0])
                except(IndexError):
                    falhado = 4
                    print('help'+str(falhado))
                
                if falhado < 4 and i%15==0:
                    final = cv2.arrowedLine(datum.cvOutputData,(int(posi[0][0]),\
                                                                    int(posi[0][1])),\
                                                                    (int(targetx),int(targety)),(0, 255, 221), thickness=3)
                    Result = coords2Quadrante((targetx,targety),final)
                    
            
                    fim = [posi[0][0],posi[0][1],\
                                       posi[17][0],posi[17][1],\
                                       posi[18][0],posi[18][1],\
                                       posi[15][0],posi[15][1],\
                                       posi[16][0],posi[16][1],\
                                       posi[4][0],posi[4][1],\
                                       posi[7][0],posi[7][1],\
                                       Result[0]]
                    
                    if (sys.argv[3]) == "y":
                        cv2.imshow('image',final)
                    cv2.waitKey(50)
        
                    
                
                
                    new_fim = [-1 if x==0 else x for x in fim[:-1]]+conta[0]+conta[1]+[fim[-1]]
                    for x in new_fim:
                        if x == -1:
                            falhado += 1
    
                if falhado < 3 and i%15==0:
                    cv2.imwrite('lixo/'+sys.argv[1][:-4]+'_'+sys.argv[2]+str(i)+'_'+str(m)+'.png',Result[1])
                    f= open('lixo/'+sys.argv[1][:-4]+'_'+sys.argv[2]+str(i)+'_'+str(m)+'.txt',"w+")
                    f.writelines(str(new_fim))
                    f.close()  
                    if (sys.argv[3]) == "y":
                        cv2.imshow('image',final)
                    cv2.waitKey(50)
    
                m += 1
            i = i+1
        else:
            break
     
    cap.release()
    cv2.destroyAllWindows()
 
master(sys.argv[1],sys.argv[2])