import cv2
import numpy
from face_recognition.FaceDetector import FaceDetector
from face_recognition.EmotionClassifier import EmotionClassifier
from face_recognition.Gender_classifier import GenderClassifier

def drawFacesImg(image):

    if False:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]
        return image

    Width = image.shape[1]
    Height = image.shape[0]


    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    index = 0
    for out in outs:
        for detection in out:
            index = index + 1
            scores = detection[5:]
            class_id = numpy.argmax(scores)
            confidence = scores[class_id]
            #if(detection[5] > 0.4):
                #print(str(index) + ' -- ' + str(detection[5]))
                #print(class_id)
                #print(confidence)
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    #print(boxes)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        #print(boxes[i])
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    return image


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    # label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    # cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)





face_detection = cv2.CascadeClassifier()

ec = EmotionClassifier()
go = GenderClassifier()

scale = 0.00392
cap = cv2.VideoCapture(0)
net = cv2.dnn.readNet('new_face_det.weights','new_face_det.cfg')

with open('classes.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = numpy.random.uniform(0, 255, size=(len(classes), 3))
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
fd = FaceDetector()
while True:
    ret,frame = cap.read()

    # img = cv2.imread('Two_faces.png')
    frame = fd.drawFacesImg(frame)

    frame = fd.drawFacesImg(frame)
    frame = ec.classify_emotion(frame,fd.boxes)
    frame2 = go.classify_gender(frame,fd.boxes)
    cv2.imshow('Face with emotion', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#img = cv2.imread('Screenshot_27.png')
#img = drawFacesImg(img)
#cv2.imshow('Two_faces.png',img)
cv2.waitKey()
cap.release()
cv2.destroyAllWindows()