import time
import cv2
import matplotlib as plt
import numpy as np

cap = cv2.VideoCapture(0)
'''address="http://192.168.0.102:8080/video"
cap.open(address)'''

ret, frame1 = cap.read()
frame1 = cv2.resize(frame1, (1280,760))



fps = cap.get(cv2.CAP_PROP_FPS)

frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)

time.sleep(int(1))
c=0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280,760))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    res = frame-frame1

    prev_y_channel = frame1[:, :, 0]
    frame_y_channel = frame[:, :, 0]

    threshold_value = 0

    mask = np.abs(frame_y_channel - prev_y_channel) > threshold_value

    res = frame.copy()
    res[:, :, 0] = np.where(mask, 0, frame_y_channel)
    res[:, : 1] = np.where(True, 128, 128)
    res[:, : 2] = np.where(True, 128, 128)
    if c==5: print(res)

    res = cv2.cvtColor(res, cv2.COLOR_YCrCb2BGR)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2YCrCb)

    detector = cv2.SimpleBlobDetector_create()

    infor = detector.detect(res)


    res = cv2.drawKeypoints((res, infor, np.array([]), (200,0,0), cv2.LINE_4))


    cv2.imshow('a0', res)
    frame1 = frame
    cv2.waitKey(int(1000))
    #cv2.destroyAllWindows()

