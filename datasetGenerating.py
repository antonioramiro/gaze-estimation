import cv2 #Open CV2
import pyyolo #YOLO
from openpose import pyopenpose as op #OpenPose
import numpy as np #Numpy
import sys #manipulating function arguments
from datetime import date #for better organizing results

def coords2Quadrante(coords,img,returnImg):
    # coords2Quadrante calculates to which quadrant (out of 16) a tuple of coordinates refers to;
    # the function's parameters include the coordinates - a tuple, *coords* -, the image object *img* (since it will)
    # base itself on the image dimensions and whether the image is or is not to be manipulated *returnImng* True or False
    # It outputs the quadrant and the image with the "calculated", if requested
        
    if returnImg == True: overlay = img.copy() # safe copy of the image to apply alpha (only if requested)
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
                else:
                    if returnImg: cv2.rectangle(overlay, (0,0), (int(w/4),int(h/4)), (229,88,191), thickness=-1) 
                    quadrante=2
                    if returnImg: cv2.rectangle(overlay, (0,int(h/4)), (int(w/4),int(h/2)), (229,88,191), thickness=-1)
            else:
                 if coords[1]<int(h/4):
                    quadrante=1
                    if returnImg: cv2.rectangle(overlay, (int(w/4),0), (int(w/2),int(h/4)), (229,88,191), thickness=-1)
                 else:
                    quadrante=3               
                    if returnImg: cv2.rectangle(overlay, (int(w/4),int(h/4)), (int(w/2),int(h/2)), (229,88,191), thickness=-1)
        else:
            if coords[0]<int(w/4): 
                if coords[1]<int((3*h)/4):
                    quadrante=8
                    if returnImg: cv2.rectangle(overlay, (0,int(h/2)), (int(w/4),int((3*h)/4)), (229,88,191), thickness=-1)
                    
                else:
                    quadrante=10
                    if returnImg: cv2.rectangle(overlay, (0,int((3*h)/4)), (int(w/4),int(h)), (229,88,191), thickness=-1)
            else:
                 if coords[1]<int((3*h)/4):
                    quadrante=9
                    if returnImg: cv2.rectangle(overlay, (int(w/4),int(h/2)), (int(w/2),int((3*h)/4)), (229,88,191), thickness=-1)
                 else:
                    quadrante=11               
                    if returnImg: cv2.rectangle(overlay, (int(w/4),int((3*h)/4)), (int(w/2),h), (229,88,191), thickness=-1)
    else:
        if coords[1]<int(h/2): 
            if coords[0]<int((3*w)/4):
                if coords[1]<int(h/4):
                    quadrante=4
                    if returnImg: cv2.rectangle(overlay, (int(w/2),0), (int((3*w)/4),int(h/4)), (229,88,191), thickness=-1)
                 
                else:
                    quadrante=6
                    if returnImg: cv2.rectangle(overlay, ((int(w/2),int(h/4))), (int((3*w)/4),int(h/2)), (229,88,191), thickness=-1)
                   
            else:
                 if coords[1]<int(h/4):
                    quadrante=5
                    if returnImg: cv2.rectangle(overlay, (int((3*w)/4),0), (int(w),int(h/4)), (229,88,191), thickness=-1)
                    
                 else:
                    quadrante=7          
                    if returnImg: cv2.rectangle(overlay, (int((3*w)/4),int(h/4)), (int(w),int(h/2)), (229,88,191), thickness=-1)
                    
        else:
            if coords[0]<int((3*w)/4): 
                if coords[1]<int((3*h)/4):
                    quadrante=12
                    if returnImg: cv2.rectangle(overlay, (int(w/2),int(h/2)), (int((3*w)/4),int((3*h)/4)), (229,88,191), thickness=-1)
                else:
                    quadrante=14
                    if returnImg: cv2.rectangle(overlay, (int(w/2),int((3*h)/4)), (int((3*w)/4),int(h)), (229,88,191), thickness=-1)
            else:
                 if coords[1]<int((3*h)/4):
                    quadrante=13
                    if returnImg: cv2.rectangle(overlay, (int((3*w)/4),int(h/2)), (int(w),int((3*h)/4)), (229,88,191), thickness=-1)
                 else:
                    quadrante=15               
                    if returnImg: cv2.rectangle(overlay, (int((3*w)/4),int((3*h)/4)), (int(w),int(h)), (229,88,191), thickness=-1)                     
    
    if returnImg:
        new = cv2.addWeighted(overlay, 0.4, img, 1 - 0.4, 0, img) #purple overlay 
        return quadrante,new #returns the quadrant and a new picture, with the underlined quadrant
    else:
        return quadrante,0 #returning just the quadrant and an empty slot (0) to spare procrssing time


def short_long(entity):
    #short_long() classify the distance of the interaction based on the kind of the object (if you hold it = short)

    distance='null'
    if entity in ['tv','pottedplant','vase','cat']:
        distance='long'
    if entity in ['chair','dining table','refrigerator','microwave','sink','bowl','apple','banana','laptop',\
                'keyboard','mouse','knife','fork','backpack']:
        distance='short'
    return distance


def main(video,VFOA,visualFeedback):
    #main function, converts a video *video*, a ground-truth visual focus of atention *VFOA* (allowing the user 
    # to choose whether she/he wants to see a visual representation X11 - Xming used) 

    resultingVector = [0,0,0,0,0,0,0,0,0,0,\
                       0,0,0,0,0,0,0,0,0,0,\
                       0,0,0,0,0,0,0,0,0,0,\
                       0,0,0,0,0,0,0,0,0,0,\
                       0,0,0,0,0,0,0] #coordinates of the head ((x,y)*5 keypoints) and hands ((x,y)*2 keypoints),
                                      #context (16 quadrants * 2 conditions (short or long))  and quadrant (1) = 47

    #initializing YOLO detector
    detector = pyyolo.YOLO("/opt/darknet/cfg/yolov3.cfg", 
                           "/opt/darknet/cfg/yolov3.weights",
                           "/opt/darknet/cfg/coco.data",
                           detection_threshold = 0.5,
                           hier_threshold = 0.5,
                           nms_threshold = 0.45)

    #initializing OpenPose
    opWrapper = op.WrapperPython()
    params = dict()
    params["model_folder"] = "/opt/openpose/models"
    opWrapper.configure(params)
    opWrapper.start()

    #inputing the video to OpenCV 
    cap = cv2.VideoCapture(video)
    
    #initializing frame counter and calculating total frames
    j = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #cycling through the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        #in order to have higher variance in data, only accepting 1 out of a sequence of 15 frames
        if j%15 != 0: 
            j += 1

        else:
            xVFOA, yVFOA = -1, -1 #initializing variables VFOA = visual focus of attention (x and y). Note that 
                                    #(-1,-1) is an impossible VFOA, so it'll trigger an error, if needed
                    
            objectDistance = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] 
                                            #initializng a variable that'll count how many close/long range
                                            #objects are in each quadrant
                                            
            maxDiagonal = -1 #aproximation: if there are several objects of the same type of the VFOA,
                                # it's assumed that the biggest one (closest to the camera) is the right
                                # one
            
            print('Frame: ' + str(j) + '/' + str(total_frames)) #sense of progress (how many frames left)

            #executing YOLO detector
            dets = detector.detect(frame) 

            #executing OpenPose
            datum = op.Datum() 
            imageToProcess = frame
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop([datum])
            frame = datum.cvOutputData

            #people counter    
            if np.cumsum(datum.poseKeypoints)[0]==255: #if there're no people OpenPose will ouput an array with only 255 in it
                peopleCounter = 0
            else:
                peopleCounter = len(datum.poseKeypoints) #else, counting how many people there are. if there were none, 
                                                         #this len() would output an error - this way, it won't and
                                                         #we're still capable of counting the people

            #only interested in scenes with 1 subject, ignore all others
            if peopleCounter != 1:            
                print('There are ' + str(peopleCounter) + ' people in this frame, thus it will be ignored.')

            else:
                keypoints = (datum.poseKeypoints[0]) #list of keypoints

                # YOLO detector, a cicle for each object
                for i, det in enumerate(dets):
                    print(f'Detection: {i}, {det}')
                    xmin, ymin, xmax, ymax = det.to_xyxy()
                    x, y= int((xmin+xmax)/2), int((ymin + ymax)/2) #center of the object
                    
                    quadrant = coords2Quadrante((x,y), frame, False)[0] #detect in which quadrant it is
                    distance = short_long(det.name) #classify the type of interaction

                    if distance =='short':  #counting the number of objects in each quadrant
                        objectDistance[0][quadrant] += 1
                    elif distance =='long':
                        objectDistance[1][quadrant] += 1

                    #visual aid
                    if visualFeedback:
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255))
                        cv2.putText(frame, str(det.name), (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0)) #name over the object
                        cv2.putText(frame, str(j)+'/'+str(total_frames), (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 0)) #frame number

                    if det.name == VFOA: #If there are more than one object of the VFOA kind, it assumes the right one to be the one  
                        thisDiagonal = (np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2))         #closest to the screen and, therefore,
                        if thisDiagonal > maxDiagonal:                                    # the one with the biggest diagonal
                            xVFOA, yVFOA = x,y
                            maxDiagonal = thisDiagonal

                if xVFOA + yVFOA == -2: #If the targeted object was not found, x and y VFOA remained with its original values (-1 and -1)
                    print('The requested target (' + VFOA + ') was not found on this frame, thus it will be ignored.')

                else: #if the VFOA was found, proceed
                    if visualFeedback: cv2.arrowedLine(frame,(int(keypoints[0][0]),int(keypoints[0][1])),(int(xVFOA),int(yVFOA)),(0, 255, 221), thickness=1)
                    quadrantVFOA = coords2Quadrante((xVFOA, yVFOA),frame,True)[0]  #quadrant of the VFOA

                    #building the result, firstly with keyposes
                    #Check https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md - Pose Output Format (BODY_25)
                    resultingVector = [keypoints[0][0],keypoints[0][1], \
                                       keypoints[17][0],keypoints[17][1],\
                                       keypoints[18][0],keypoints[18][1],\
                                       keypoints[15][0],keypoints[15][1],\
                                       keypoints[16][0],keypoints[16][1],\
                                       keypoints[4][0],keypoints[4][1],\
                                       keypoints[7][0],keypoints[7][1]]
                                       #nose, right ear, left ear, right eye, left eye, right hand, left hand

                    resultingVector = [-1 if x==0 else x for x in resultingVector] #increasing the strangeness of undetected points
                                                                                   # so that it's more evident for thge SVM to understand

                    #counting the number of keyposes that weren't detected, if there're more than 3 (3*(x,y) = 6), the data is discarded                                 
                    if resultingVector.count(-1) >= 6:  #this tolerance value can be changed          
                        print('There were a total of ' + str(int(resultingVector.count(0))/2) + ' keypoints missing. Thus, this frame will be ignored.')                 

                    else:
                        #adding context and the quadrant to the vector
                        resultingVector += objectDistance[0] + objectDistance[1] + [quadrantVFOA]

                        #outputing to a textfile, note that the chosen name is extremely tailored, manipulating expected inputs
                        # in particular, videos found in ~/source_videos/ folder 
                        f= open('dataset/' + str(date.today()) + '_' + sys.argv[1][14:-4]+'_'+sys.argv[2]+'_'+str(j)+'.txt',"w+") 
                        f.writelines(str(resultingVector))
                        f.close()   

                    if visualFeedback: cv2.imshow('cvwindow', frame) #showing the frame
                    if visualFeedback: cv2.waitKey(10) #waiting - this value can be decreased, to shorten generation times

            print('\n') #visual shell organization
            j += 1 #next frame
            
            if cv2.waitKey(1) == 27: #???
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], bool(sys.argv[3]))

