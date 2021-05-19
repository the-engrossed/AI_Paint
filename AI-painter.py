import cv2
import numpy as np
import time
import os
import HandTracking as ht

#for brush thickness declare the variable to a number you like
brush_thickness = 5

#for eraser thickness
eraser_thickness = 50

filePath = "Header"
Imglist = os.listdir(filePath)
print(Imglist)

oList = []

for imgpath in Imglist:
    image = cv2.imread(f'{filePath}/{imgpath}')
    oList.append(image)

print(len(oList))

header = oList[0]
drawing_color =(0,255,255)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(3,720)

detector = ht.handTracker(detectionConf=0.85)
prev_x, prev_y = 0, 0
img_canvas = np.zeros((480,640,3), np.uint8)

while True:
    #Import toolbox images
    success , img = cap.read()
    img = cv2.flip(img, 1) #flipping the images so that when we draw on right it draws on the right side of the camera

    #Finding the hand landmarks
    img = detector.findingHands(img)
    landmarkList = detector.findingPositions(img, draw=False)

    if len(landmarkList) != 0:

        #for the tip of index finger
        x1, y1 = landmarkList[8][1:]
        #for the tip of the middle finger
        x2, y2 = landmarkList[12][1:]

        #Checking which fingers are up

        fingers = detector.fingersUp()

        #Selecting color when two fingers are up

        if fingers[1] and fingers[2]:

            prev_x, prev_y = 0, 0

            print("Selection Mode")

            #Checking for the color
            if y1 < 125:
                if 45 < x1 < 110:
                    header = oList[0]
                    drawing_color = (0, 255, 255)
                elif 180 < x1 < 240:
                    header = oList[1]
                    drawing_color = (0,0,255)
                elif 305 < x1 < 365:
                    header = oList[2]
                    drawing_color = (255, 0, 0)
                elif 425 < x1 < 490:
                    header = oList[3]
                    drawing_color = (0, 255, 0)
                elif 535 < x1 < 610:
                    header = oList[4]
                    drawing_color = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), drawing_color, cv2.FILLED)

        #Draw if Index finger is up

        if fingers[1] and fingers[2]== False:
            cv2.circle(img, (x1,y1), 15,drawing_color, cv2.FILLED)
            print("drawing mode")
            #to start drawing from the point you are at on the screen
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x1, y1
                # cv2.line(img,(prev_x, prev_y),(x1,y1), drawing_color,brush_thickness)

            #thickness of eraser
            if drawing_color == (0,0,0):
                cv2.line(img, (prev_x, prev_y), (x1, y1), drawing_color, eraser_thickness)
                cv2.line(img_canvas, (prev_x, prev_y), (x1, y1), drawing_color, eraser_thickness)

            else:
                #drawing the lines
                cv2.line(img, (prev_x, prev_y), (x1, y1), drawing_color, brush_thickness)
                cv2.line(img_canvas, (prev_x, prev_y), (x1, y1), drawing_color, brush_thickness)

            prev_x, prev_y = x1, y1

    #masking the canvas with the original image
    imgGray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,img_canvas)



#Setting the toolbox image on top of the camera screen
    img[0:125, 0:640] = header
    # #adding the original image and the canvas
    # img = cv2.addWeighted(img, 0.5, img_canvas, 0.5,0)
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", img_canvas)
    # cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)
