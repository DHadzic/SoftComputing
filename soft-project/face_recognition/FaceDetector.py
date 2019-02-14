import cv2
import numpy

class FaceDetector:
    def __init__(self):
        self.scale = 0.00392
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.net = cv2.dnn.readNet('new_face_det.weights','new_face_det.cfg')

    def drawFacesImg(self,image):

        if False:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = image[y:y + h, x:x + w]
            return image

        Width = image.shape[1]
        Height = image.shape[0]


        blob = cv2.dnn.blobFromImage(image, self.scale, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        outs = self.net.forward(self.get_output_layers())

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
            self.draw_bounding_box(image, confidences[i], round(x), round(y), round(x + w), round(y + h))

        return image


    def get_output_layers(self):
        layer_names = self.net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        return output_layers

    def draw_bounding_box(self,img,confidence, x, y, x_plus_w, y_plus_h):

        #label = str(classes[class_id])

        #color = COLORS[class_id]

        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (212, 118, 211), 2)

        #cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

