from ultralytics import YOLO
import cvzone
import cv2
import math





cap = cv2.VideoCapture(0)
model = YOLO('fire.pt')
address="http://192.168.0.102:8080/video"
cap.open(address)

fps = cap.get(cv2.CAP_PROP_FPS)



classnames = ['fire']

while True:
    ret,frame = cap.read()
    #frame = cv2.resize(frame,(640,480))
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    result = model(frame,stream=True)

    
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 40:
                x1,y1,x2,y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5,thickness=2)




    cv2.imshow('frame',frame)
    cv2.waitKey(int(1000/fps))