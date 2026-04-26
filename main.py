import cv2
import os
import numpy as np
from ultralytics import YOLO

cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

AGE_BUCKETS = ["(0-2)","(4-6)","(8-12)","(15-20)","(20-24)",
               "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
BASE = os.path.dirname(os.path.abspath(__file__))
age_net = cv2.dnn.readNet(
    os.path.join(BASE, "age_net.caffemodel"),
    os.path.join(BASE, "age_deploy.prototxt")
)
MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
video_capture = cv2.VideoCapture(0)
yolo = YOLO("yolov8s.pt")

while True:
    ret, frames = video_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face_roi = frames[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MODEL_MEAN)
        age_net.setInput(blob)
        preds = age_net.forward()[0]
        age = AGE_BUCKETS[preds.argmax()]

        cv2.putText(frames, f"Age: {age}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    results = yolo(frames, verbose=False)
    for result in results:
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = yolo.names[cls]
                cv2.rectangle(frames, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frames, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Video', frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()