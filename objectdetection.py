#Verison: 0.1.8
#Author: Rhn15001, Rasmus Hamren
#Applying UAVs to Support the Safety in Autonomous Operated Open Surface Mines

#Edge-detection is inspired by reference:
#A.Rosebrock,“Targetacquired:Findingtargetsindroneandquad-
#coptervideostreamsusingpythonandopencv,”, PyImageSearch,
#accessed:2021-01-04.[Online].
#Available: https://tinyurl.com/y6kd3tsy

#Object detection, YOLO, is inspired by reference:
#Abdulkhadeer0200,
#“Real-time-object-detection-using-yolo,”
#https://tinyurl.com/y34qtepv,
#commit= 5d77ca5c23928df4f851471ad90ee413728f1c2b, 2019, accessed:  2021-01-04.

import cv2
import numpy as np
import sys
import imutils

imagename = sys.argv[1]
yoloV = sys.argv[2]

#Variables
c2 = 0
a = 1
xo = 0; xsave = 0; xsaveV = 0
yo = 0; ysave = 0; wsave = 0; ysaveV = 0; wsaveV = 0
diff = 2
yxdiff = 1
yydiff = 1
t = 0
line_thickness = 2
Multiplier = 0.5
OldCentrumX = 0
OldCentrumY = 0
add = 0
loop_stopper = loop_stopper2 = 0

#Load YOLO

# Original yolov4
if(yoloV == "v4"):
    net = cv2.dnn.readNet("yolov4.weights","yolov4.cfg")

# Original yolov3
if(yoloV == "v3-tiny"):
    net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg")

# Tiny yolov4
if(yoloV == "v4-tiny"):
    net = cv2.dnn.readNet("yolov4-tiny.weights","yolov4-tiny.cfg")

# Tiny yolov3
if(yoloV != "v3-tiny" and yoloV != "v4" and yoloV != "v4-tiny"):
    print("Enter value after video name for Weight: v4, v3-tiny")

classes = []

with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

#print(classes)

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0,255,size=(len(classes),3))

#loading image
font = cv2.FONT_HERSHEY_PLAIN
frame_id = 0
cap = cv2.VideoCapture(imagename)

#FRAME TO START ON

while True:
    (grabbed, frame) = cap.read()

#JUMPOVER PICTURES#
    if(yoloV == "v4-tiny" or yoloV == "v3-tiny"):
        for x in range(0, 1):
            _,frame= cap.read()

    if not (yoloV == "v4-tiny" or yoloV == "v3-tiny"):
        for x in range(0, 10):
            _,frame= cap.read()

#Flip if want to
    #frame = cv2.flip(frame,-1) #FlipTest

    if not grabbed:
        break
        print("ERROR WHILE GRAB")

    ##CHANGE COLOR SPECTRUM

    b, g, r = cv2.split(frame)
    frame = cv2.merge((b, g, r))

    #detecting objects
    blob = cv2.dnn.blobFromImage(frame,0.00392,(416,416),
    (0,0,0),True,crop=False) #reduce 416 to 320

    net.setInput(blob)
    outs = net.forward(outputlayers)

    #Showing info on screen/ get confidence score of algorithm in
    #detecting an object in blob
    class_ids=[]
    confidences=[]
    boxes=[]

    height,width,channels = frame.shape
    #print(frame.shape[1])
    #print(frame.shape[0])

    #########EDGE FIXES############
    # convert the frame to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame
    blurred = cv2.GaussianBlur(gray, (3, 3), 5)
    edged = cv2.Canny(blurred, 200, 500)
    #find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)

    #cv2.imshow("Edged",edged)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:

                #onject detected
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #rectangle co-ordinaters
                x=int(center_x - w/2)
                y=int(center_y - h/2)

                boxes.append([x,y,w,h]) #put all rectangle areas

                #how confidence was that object detected and show that percentage
                confidences.append(float(confidence))

                class_ids.append(class_id) #name of the object tha was detected

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]

            if(yoloV == "v4" or yoloV == "v3"):
                acc = 0.70

            if not (yoloV == "v4" or yoloV == "v3"):
                acc = 0.40

            if(round(confidence,2) >= acc):
                if(label == "cat"):
                    cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                    cv2.putText(frame,label+" "+str(round(confidence,2)),
                    (x,y+30),font,1,(255,0,0),2)

                if not (label == "cat"):
                    cv2.circle(frame,(center_x,center_y),10,(0,255,0),2)
                    print(label)
                    cv2.putText(frame," UNDEFINED OBJECT; BE CAREFUL ",
                    (x,y+30),font,1,(255,0,0),2)

    #camangle compared to north

    #CircleBufferForSafety

    cv2.imshow("Image",frame)

    key = cv2.waitKey(1) #wait 1ms the loop will start again

    if key == 27: #esc key stops the process
        cv2.imwrite("ResultDot.png",im)
        break;

cap.release()
cv2.destroyAllWindows()
