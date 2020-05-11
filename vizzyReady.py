#Python libs
import numpy as np
import sys, time

#Image manipulation and detection
import cv2 
import pyyolo 
from openpose import pyopenpose as op

#classifier
from joblib import dump, load
from scipy import stats
from sklearn.svm import SVC
from sklearn import svm

#ros
import roslib
import rospy
# Ros Messages
from sensor_msgs.msg import CompressedImage

VERBOSE = False 

def __init__(self):
    '''Initialize ros publisher, ros subscriber'''
    # topic where we publish
    self.image_pub = rospy.Publisher("/output/image_raw/compressed", CompressedImage)
    # self.bridge = CvBridge()

    # subscribed Topic
    self.subscriber = rospy.Subscriber("/camera/image/compressed",
        CompressedImage, self.callback,  queue_size = 1)

    
    #adicionei o que está abaixo. adicionei o 'self.' como estava nas outras coisas - ja nao sei trabalhar com classes (?)

    #initializing YOLO detector
    self.detector = pyyolo.YOLO("/opt/darknet/cfg/yolov3.cfg", 
                           "/opt/darknet/cfg/yolov3.weights",
                           "/opt/darknet/cfg/coco.data",
                           detection_threshold = 0.5,
                           hier_threshold = 0.5,
                           nms_threshold = 0.45)

    #initializing OpenPose
    self.opWrapper = op.WrapperPython()
    self.params = dict()
    self.params["model_folder"] = "/opt/openpose/models"
    self.opWrapper.configure(params)
    self.opWrapper.start()

      #loading the classifier
    self.clf = joblib.load(sys.argv[1])

    if VERBOSE :
        print "subscribed to /camera/image/compressed"

    def strToBool(string):
        if string == 'True':
            return True
        elif string == 'False':
            return False

    def paintImage(quadrant,img):            
        overlay = img.copy() # safe copy of the image to apply alpha (only if requested)
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

        if quadrant == 0: cv2.rectangle(overlay, (0,0), (int(w/4),int(h/4)), (229,88,191), thickness=-1)
        elif quadrant == 1: cv2.rectangle(overlay, (int(w/4),0), (int(w/2),int(h/4)), (229,88,191), thickness=-1)
        elif quadrant == 2: cv2.rectangle(overlay, (0,int(h/4)), (int(w/4),int(h/2)), (229,88,191), thickness=-1)
        elif quadrant == 3: cv2.rectangle(overlay, (int(w/4),int(h/4)), (int(w/2),int(h/2)), (229,88,191), thickness=-1)
        elif quadrant == 4: cv2.rectangle(overlay, (int(w/2),0), (int((3*w)/4),int(h/4)), (229,88,191), thickness=-1)
        elif quadrant == 5: cv2.rectangle(overlay, (int((3*w)/4),0), (int(w),int(h/4)), (229,88,191), thickness=-1)
        elif quadrant == 6: cv2.rectangle(overlay, ((int(w/2),int(h/4))), (int((3*w)/4),int(h/2)), (229,88,191), thickness=-1)
        elif quadrant == 7: cv2.rectangle(overlay, (int((3*w)/4),int(h/4)), (int(w),int(h/2)), (229,88,191), thickness=-1)
        elif quadrant == 8: cv2.rectangle(overlay, (0,int(h/2)), (int(w/4),int((3*h)/4)), (229,88,191), thickness=-1)
        elif quadrant == 9: cv2.rectangle(overlay, (int(w/4),int(h/2)), (int(w/2),int((3*h)/4)), (229,88,191), thickness=-1)
        elif quadrant == 10: cv2.rectangle(overlay, (0,int((3*h)/4)), (int(w/4),int(h)), (229,88,191), thickness=-1)
        elif quadrant == 11: cv2.rectangle(overlay, (int(w/4),int((3*h)/4)), (int(w/2),h), (229,88,191), thickness=-1)
        elif quadrant == 12: cv2.rectangle(overlay, (int(w/2),int(h/2)), (int((3*w)/4),int((3*h)/4)), (229,88,191), thickness=-1)
        elif quadrant == 13: cv2.rectangle(overlay, (int((3*w)/4),int(h/2)), (int(w),int((3*h)/4)), (229,88,191), thickness=-1)
        elif quadrant == 14: cv2.rectangle(overlay, (int(w/2),int((3*h)/4)), (int((3*w)/4),int(h)), (229,88,191), thickness=-1)
        elif quadrant == 15: cv2.rectangle(overlay, (int((3*w)/4),int((3*h)/4)), (int(w),int(h)), (229,88,191), thickness=-1)

        
        new = cv2.addWeighted(overlay, 0.4, img, 1 - 0.4, 0, img) #purple overlay 
        return new #returns the quadrant and a new picture, with the underlined quadrant

    def short_long(entity):
        #short_long() classify the distance of the interaction based on the kind of the object (if you hold it = short)

        distance='null'
        if entity in ['tv','pottedplant','vase','cat','bowl','clock','toilet','chair','bench','couch']:
            distance='long'
        if entity in ['dining table','refrigerator','microwave','sink','apple','banana','laptop',\
                    'keyboard','mouse','knife','fork','backpack','oven','toaster']:
            distance='short'
        return distance

#copiado de novo
def callback(self, ros_data):
    '''Callback function of subscribed topic. 
    Here images get converted and features detected'''
    if VERBOSE :
        print 'received image of type: "%s"' % ros_data.format

    #### direct conversion to CV2 ####
    np_arr = np.fromstring(ros_data.data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)


    #resize frame - meti o que está a partir daqui (!)
    frame = cv2.resize(frame, (640,352))

    xVFOA, yVFOA = -1, -1 #initializing variables VFOA = visual focus of attention (x and y). Note that 
                                    #(-1,-1) is an impossible VFOA, so it'll trigger an error, if needed
                    
    objectDistance = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] 
                                            #initializng a variable that'll count how many close/long range
                                            #objects are in each quadrant
                                            
    maxDiagonal = -1 #aproximation: if there are several objects of the same type of the VFOA,
                                # it's assumed that the biggest one (closest to the camera) is the right
                                # on

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
            #building the result, firstly with keyposes
            #Check https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md - Pose Output Format (BODY_25)
            resultingVector = [keypoints[0][0],keypoints[0][1], \
                                keypoints[1][0],keypoints[1][1], \
                                keypoints[17][0],keypoints[17][1],\
                                keypoints[18][0],keypoints[18][1],\
                                keypoints[15][0],keypoints[15][1],\
                                keypoints[16][0],keypoints[16][1],\
                                keypoints[4][0],keypoints[4][1],\
                                keypoints[7][0],keypoints[7][1]]
                                #nose, neck, right ear, left ear, right eye, left eye, right hand, left hand

            resultingVector = [-1 if x==0 else x for x in resultingVector] #increasing the strangeness of undetected points
                                                                            # so that it's more evident for thge SVM to understand

            #counting the number of keyposes that weren't detected, if there're more than 3 (3*(x,y) = 6), the data is discarded                                 
            if (resultingVector[8] + resultingVector[9] != -2) and resultingVector.count(-1) >= 6:  #this tolerance value can be changed          
                print('There were a total of ' + str(int(resultingVector.count(-1))/2) + ' keypoints missing. Thus, this frame will be ignored.')                 

            else:
                #adding context and the quadrant to the vector
                resultingVector += objectDistance[0] + objectDistance[1]

                #estimating VFOA
                quadrantVFOA = clf.predict(resultingVector)
                print('The ' + VFOA + 'is thought to be in quadrant ' + str(quadrantVFOA) + '.')
                print('\n') #visual shell organization

            if visualFeedback:
                frame = paintImage(frame)
                cv2.imshow('cvwindow', frame) #showing the frame
            if visualFeedback: cv2.waitKey(3) #waiting - this value can be decreased, to shorten generation times
   
            if cv2.waitKey(1) == 27: 
                break
    
        cv2.destroyAllWindows()

def main(args):
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1])

