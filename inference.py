import torch
import cv2
import numpy as np

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp15/weights/best.pt')

# frame
cap = cv2.VideoCapture('pexels-taryn-elliott-5548301.mp4')
img = 'zidane.jpg'

# Inference
model.iou = 0.1
model.conf = 0.5

while True:
    ret,frame1 = cap.read()
    results = model(frame1)
    #print(results)
    # Results
    count = 0
    for box in results.xyxy[0]: 
            if box[5]==0:   
                count = count + 1
    # print(frame1.shape[1])
    frame1_output = np.squeeze(results.render())
    cv2.putText(frame1_output, "FISH COUNTER: " +str(count), (18,36),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    cv2.imshow('YOLO', frame1_output)
    cv2.waitKey(1)
    #results.pandas().xyxy[0]