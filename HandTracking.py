import cv2
import os
import mediapipe as mp
import time

class handTracker():
    def __init__(self, mode=False, max_no_hands=2, detectionConf= 0.5, trackingConf= 0.5 ):
        self.mode = mode
        self.max_no_hands = max_no_hands
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_no_hands, self.detectionConf, self.trackingConf)
        self.mpdraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findingHands(self, img, draw= True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
        return img

    def findingPositions(self, img, HandNo=0, draw= True):
        self.landmarkList = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[HandNo]

            for id, lm in enumerate(my_hand.landmark):
                # print(id,lm)
                height, width, channels = img.shape
                center_x, center_y = int(lm.x * width), int(lm.y * height)
                # print(id, center_x, center_y)
                self.landmarkList.append([id, center_x, center_y])
                if draw:
                    cv2.circle(img, (center_x, center_y), 10, (255, 0, 255), cv2.FILLED)

        return self.landmarkList

    def fingersUp(self):
        fingers = []

        #check if thumb is up
        if self.landmarkList[self.tipIds[0]][1] < self.landmarkList[self.tipIds[0]-1][1]:
            fingers.append(1)

        else:
            fingers.append(0)

        #for other four fingers
        for id in range(1,5):
            if self.landmarkList[self.tipIds[id]][2] < self.landmarkList[self.tipIds[id]-2][2]:
                fingers.append(1)

            else:
                fingers.append(0)

        return fingers


def main():
    prevTime = 0
    currTime = 0
    detector = handTracker()

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        img = detector.findingHands(img)
        landmarkList = detector.findingPositions(img)
        if len(landmarkList) != 0:
            print(landmarkList[4])

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__=="__main__":
    main()
