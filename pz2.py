import torch
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from scipy.spatial import distance

model = torch.hub.load('ultralytics/yolov5', 'yolov5x').cuda()
model_2  = MTCNN(keep_all=True)
cap = cv2.VideoCapture(0)
cup = False

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]),  (0, 0, 255), 10)
    # Our operations on the frame come here
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = model(frame.copy())
    obj = gray.pandas().xyxy
    boxes, pt = model_2.detect(frame.copy())
    if not isinstance(boxes, type(None)):
        if len(boxes) > 0:
            boxes = boxes[0]
        if len(obj) > 0:
            for i in obj:
                for j in zip(i['xmin'], i['ymin'], i['xmax'], i['ymax'], i['name']):
                    if j[4] == 'cup':
                        cup_min, cup_max = (int(j[0]), int(j[1])), (int(j[2]), int(j[3]))
                        frame = cv2.rectangle(frame, cup_min, cup_max,  (0, 255, 255), 5)
                        cup_x, cup_y = int((cup_min[0] + cup_max[0])/2), int((cup_min[1] + cup_max[1])/2)
                        frame = cv2.putText(frame, 'Cup', (cup_x, cup_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (255, 0, 0), 2, cv2.LINE_AA)
                        cup = True
        if len(boxes) > 0:
            face_min, face_max = (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3]))
            face_x, face_y = int((face_min[0] + face_max[0])/2), int((face_min[1] + face_max[1])/2)
            frame = cv2.rectangle(frame, face_min, face_max, (255, 255, 255), 5)
            frame = cv2.putText(frame, 'Face', (face_x, face_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (255, 0, 0), 2, cv2.LINE_AA)
        if cup:
            if distance.euclidean((face_x, face_y), (cup_x, cup_y)) <= abs(cup_min[1] - cup_max[1]):
                frame = cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]),  (0, 255, 0), 5)
        else:
            frame = cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]),  (0, 0, 255), 10)
            # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
