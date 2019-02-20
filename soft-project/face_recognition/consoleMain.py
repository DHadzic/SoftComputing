import cv2
import numpy
from FaceDetector import FaceDetector
from EmotionClassifier import EmotionClassifier
from Gender_classifier import GenderClassifier
import time


def edit_frame(frame, emotions, genders, boxes):
    for i in range(0, len(boxes)):
        x, y, x_plus_w, y_plus_h = round(boxes[i][0]), round(boxes[i][1]), round(boxes[i][0] + boxes[i][2]), round(boxes[i][1] + boxes[i][3])

        cv2.rectangle(frame, (x, y), (x_plus_w, y_plus_h), (212, 118, 211), 2)
        if len(emotions) > i:
            cv2.putText(frame, emotions[i], (int(x), int(y_plus_h) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        if len(genders) > i:
            cv2.putText(frame, genders[i], (int(x) + 55, int(y_plus_h) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        time.sleep(1)

    return frame


face_detection = cv2.CascadeClassifier()

ec = EmotionClassifier()
gc = GenderClassifier()

scale = 0.00392
cap = cv2.VideoCapture(0)
net = cv2.dnn.readNet('new_face_det.weights','new_face_det.cfg')

with open('classes.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = numpy.random.uniform(0, 255, size=(len(classes), 3))
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
fd = FaceDetector()
# while True:
#     ret,frame = cap.read()

frame = cv2.imread('main_928px.jpg')
# frame = fd.drawFacesImg(frame)

# frame = fd.drawFacesImg(frame)
# frame = ec.classify_emotion(frame,fd.boxes)
# frame2 = go.classify_gender(frame,fd.boxes)

fd.drawFacesImg(frame)
emotions = ec.classify_emotion(frame, fd.boxes)
genders = gc.classify_gender(frame, fd.boxes)
frame = edit_frame(frame, emotions, genders, fd.boxes)
cv2.imshow('Face with emotion', frame)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


#img = cv2.imread('Screenshot_27.png')
#img = drawFacesImg(img)
#cv2.imshow('Two_faces.png',img)
cv2.waitKey()
cap.release()
cv2.destroyAllWindows()