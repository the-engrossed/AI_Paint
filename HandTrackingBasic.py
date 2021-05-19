import cv2
import os
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                # print(id,lm)
                height, width, channels = img.shape
                center_x, center_y = int(lm.x*width), int(lm.y*height)
                print(id, center_x, center_y)
                if id == 0:
                    cv2.circle(img, (center_x,center_y), 10, (255,0,255), cv2.FILLED)

            mpdraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime= currTime

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
