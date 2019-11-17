#Importing packages:
import cv2 #image manipulation
import sys #manipulating function arguments
import numpy as np #eventual mathemathic operations
from pydarknet import Detector, Image #openpose
import os #operating system interactions
dir_path = os.path.dirname(os.path.realpath(__file__))
from openpose import pyopenpose as op #open pose
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# encoding: utf-8

# coords2Quadrante calculates to which quadrant (out of 16) a tuple of coordinates refers
# the function's parameters include the coordinates - a tuple - and the image object
# It outputs the quadrant and the image with the calculated quadrant overlined
def coords2Quadrante(coords,img,returnImg):
    
    if returnImg == 'y': overlay = img.copy() # safe copy of the image to apply alpha (only if requested)
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
                    if returnImg == 'y': cv2.rectangle(overlay, (0,0), (int(w/4),int(h/4)), (229,88,191), thickness=-1) 
                else:
                    quadrante=2
                    if returnImg == 'y': cv2.rectangle(overlay, (0,int(h/4)), (int(w/4),int(h/2)), (229,88,191), thickness=-1)
            else:
                 if coords[1]<int(h/4):
                    quadrante=1
                    if returnImg == 'y': cv2.rectangle(overlay, (int(w/4),0), (int(w/2),int(h/4)), (229,88,191), thickness=-1)
                 else:
                    quadrante=3               
                    if returnImg == 'y': cv2.rectangle(overlay, (int(w/4),int(h/4)), (int(w/2),int(h/2)), (229,88,191), thickness=-1)
        else:
            if coords[0]<int(w/4): 
                if coords[1]<int((3*h)/4):
                    quadrante=8
                    if returnImg == 'y': cv2.rectangle(overlay, (0,int(h/2)), (int(w/4),int((3*h)/4)), (229,88,191), thickness=-1)
                    
                else:
                    quadrante=10
                    if returnImg == 'y': cv2.rectangle(overlay, (0,int((3*h)/4)), (int(w/4),int(h)), (229,88,191), thickness=-1)
            else:
                 if coords[1]<int((3*h)/4):
                    quadrante=9
                    if returnImg == 'y': cv2.rectangle(overlay, (int(w/4),int(h/2)), (int(w/2),int((3*h)/4)), (229,88,191), thickness=-1)
                 else:
                    quadrante=11               
                    if returnImg == 'y': cv2.rectangle(overlay, (int(w/4),int((3*h)/4)), (int(w/2),h), (229,88,191), thickness=-1)
    else:
        if coords[1]<int(h/2): 
            if coords[0]<int((3*w)/4):
                if coords[1]<int(h/4):
                    quadrante=4
                    if returnImg == 'y': cv2.rectangle(overlay, (int(w/2),0), (int((3*w)/4),int(h/4)), (229,88,191), thickness=-1)
                 
                else:
                    quadrante=6
                    if returnImg == 'y': cv2.rectangle(overlay, ((int(w/2),int(h/4))), (int((3*w)/4),int(h/2)), (229,88,191), thickness=-1)
                   
            else:
                 if coords[1]<int(h/4):
                    quadrante=5
                    if returnImg == 'y': cv2.rectangle(overlay, (int((3*w)/4),0), (int(w),int(h/4)), (229,88,191), thickness=-1)
                    
                 else:
                    quadrante=7          
                    if returnImg == 'y': cv2.rectangle(overlay, (int((3*w)/4),int(h/4)), (int(w),int(h/2)), (229,88,191), thickness=-1)
                    
        else:
            if coords[0]<int((3*w)/4): 
                if coords[1]<int((3*h)/4):
                    quadrante=12
                    if returnImg == 'y': cv2.rectangle(overlay, (int(w/2),int(h/2)), (int((3*w)/4),int((3*h)/4)), (229,88,191), thickness=-1)
                else:
                    quadrante=14
                    if returnImg == 'y': cv2.rectangle(overlay, (int(w/2),int((3*h)/4)), (int((3*w)/4),int(h)), (229,88,191), thickness=-1)
            else:
                 if coords[1]<int((3*h)/4):
                    quadrante=13
                    if returnImg == 'y': cv2.rectangle(overlay, (int((3*w)/4),int(h/2)), (int(w),int((3*h)/4)), (229,88,191), thickness=-1)
                 else:
                    quadrante=15               
                    if returnImg == 'y': cv2.rectangle(overlay, (int((3*w)/4),int((3*h)/4)), (int(w),int(h)), (229,88,191), thickness=-1)                    
            
    if returnImg == 'y': new = cv2.addWeighted(overlay, 0.4, img, 1 - 0.4, 0, img) #purple overlay 
    
    
    if returnImg == 'y':
        return quadrante,new #returns the quadrant and a new picture, with the underlined quadrant
    else:
        return quadrante #returning just the quadrant, to spare procrssing time


# contextual information - classifying an object in long or short range interaction
def short_long(entity):
    distance='null'
    if entity in ['tv','pottedplant','vase','cat']:
        distance='long'
    if entity in ['chair','dining table','refrigerator','microwave','sink','bowl','apple','banana','laptop',\
                'keyboard','mouse','knife','fork','backpack']:
        distance='short'
    return distance

# master function, receives a video and the ground truth object (Visual Focus of Atention)
def master(video,VFOA):
    cap = cv2.VideoCapture(video)
    #Initializing openPose, specifying directories
    params = dict()
    params["model_folder"] = "/home/sims/repositories/openpose/models/" 
      
    #I'm not sure this part is crucial
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
     
    #OpenPose initializing 
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
    #Initializing YOLO
     
    net = Detector(bytes("/home/sims/repositories/YOLO3-4-Py/cfg/yolov3.cfg", encoding="utf-8"), \
    bytes("/home/sims/repositories/YOLO3-4-Py/weights/yolov3.weights", encoding="utf-8"), 0, 
    bytes("/home/sims/repositories/YOLO3-4-Py/cfg/coco.data",encoding="utf-8"))  
     
    i = 0 #frame counter
    while(True):
        ret, frame = cap.read() 
        if ret: #ret = correctly read
            print(i) 
            
            frame = cv2.resize(frame,(1280 ,720)) #converting each frame to a lower resolution
            fliped_frame = cv2.flip(frame,1) #to increase sample size, simetric frame will also be considered
            tolerance = 0 #initializing variable. it will allow for a certain error treshold
            
            images = [frame,fliped_frame] #normal and fliped frame
            isFlipped = 'orginial'
            for img in images:  #run once with the image in its normal orientation, and once while fliped
                
                if (sys.argv[3]) == "y": #display starting image
                    cv2.imshow('Current Frame',img)
                    cv2.waitKey(25) #wait 25 ms to show next image (if available)
            

                xVFOA, yVFOA = -1, -1 #initializing variables VFOA = visual focus of attention (x and y). Note that 
                                 #(-1,-1) is an impossible VFOA, so will trigger the error, if needed
                
                #initializing and running YOLO object detetion   , which will give contextual information 
                dark_frame = Image(img) 
                results = net.detect(dark_frame) #results is a list of lists - one list for each object, containing: its name,
                                                 #the amount of trust in the classification, and the bounds.
            
                objectDistance = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] #initializng a variable 
                                            #that will count how many objects are in each quadrant and whether they are of close
                                            #or long range interaction, to complemnt contextual information
                maxheight = -1
                for cat, score, bounds in results: 
                    
                    x, y, w, h = bounds # x,y - center of the object, w,h = width and height
                    cv2.rectangle(img, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(0,0,255), thickness=2) #rectangle over the object
                    cv2.putText(img, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0)) #name over the object
                 
                    distance = short_long(str(cat.decode("utf-8"))) #check whether the object is short or long range 
                    quadrant = coords2Quadrante((x,y),img,'n') #see in which quadrant the object is
                
                    if distance =='short':  #counting the number of objects in each quadrant
                        objectDistance[0][quadrant] += 1
                    elif distance =='long':
                        objectDistance[1][quadrant] += 1
                    
                    if str(cat.decode("utf-8")) == VFOA:    #if certain object is the ground truth object to which the person is looking at,                                                       
                        if h > maxheight:                   # save its coordinates. If there are several "focus" objects, choose the closest
                            maxheight = h                   # - the one with the maximum height - 
                            xVFOA = x
                            yVFOA = y
                            
                cv2.circle(img,(int(xVFOA),int(yVFOA)), 3,(255,255,255), thickness=5)    #circling the VFOA on the image          
              
                del dark_frame #cleaning network
            
                if xVFOA + yVFOA == -2: #triggering error
                    tolerance = 4
        
                datum = op.Datum() #running OpenPose - (would like to make OpenPose run on a clean image - not one full of rectangles from YOLO)
                imageToProcess = frame                              #and add the skeletons to the image after YOLO analyses it
                datum.cvInputData = imageToProcess
                opWrapper.emplaceAndPop([datum]) 
            
                
                try: #will only run if only one person was detected in shot, else, will skip this frame        
                    keypoints = (datum.poseKeypoints[0])
                except(IndexError):
                    tolerance = 4  #trigger error 
                
                if tolerance < 4 and i%15==0: #if tolerable, outputing results. i%15 frame is to only take four sample frames out of each second of video
                                              #I'd like to move this filter upwards in order to spare processing power (used by Open Pose and YOLO)
                                              #I wasn't able to do this in previous testing. Will try again.                   
                                              
                    resultingImg = cv2.arrowedLine(datum.cvOutputData,(int(keypoints[0][0]),int(keypoints[0][1])),\
                                                                    (int(xVFOA),int(yVFOA)),(0, 255, 221), thickness=3) #draws an arrow, from the eyes to the VFOA
            
                    resultingVector = [keypoints[0][0],keypoints[0][1],\
                                       keypoints[17][0],keypoints[17][1],\
                                       keypoints[18][0],keypoints[18][1],\
                                       keypoints[15][0],keypoints[15][1],\
                                       keypoints[16][0],keypoints[16][1],\
                                       keypoints[4][0],keypoints[4][1],\
                                       keypoints[7][0],keypoints[7][1],\
                                       coords2Quadrante((xVFOA,yVFOA),resultingImg,'n')] #the resulting vector, with: face coordinates, 
                                                        #hand coordinates and the quadrant
        
                
                    resultingVector = [-1 if x==0 else x for x in resultingVector[:-1]]+objectDistance[0]+objectDistance[1]+[resultingVector[-1]]
                                    #adding the object distance and signing keypoints that weren't found
                    
                    for x in resultingVector: #counting the keypoints that weren't detected
                        if x == -1:
                            tolerance += 1
                            
                            
                            
                if tolerance < 4 and i%15==0: #if tolerance is bellow a certain treshold
                    cv2.imwrite('lixo/'+sys.argv[1][:-4]+'_'+sys.argv[2]+str(i)+'_'+isFlipped+'.png',coords2Quadrante((xVFOA,yVFOA),resultingImg,'y')[1])
                    f= open('lixo/'+sys.argv[1][:-4]+'_'+sys.argv[2]+str(i)+'_'+isFlipped+'.txt',"w+")
                    f.writelines(str(resultingVector))
                    f.close()  
                    if (sys.argv[3]) == "y":
                        cv2.imshow('image',resultingImg)
                    cv2.waitKey(50)

                isFlipped = 'flipped'
            i = i+1
        else:
            break
     
    cap.release()
    cv2.destroyAllWindows()
 
master(sys.argv[1],sys.argv[2])