import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2


class GenderClassifier:

    def __init__(self):

        # self.model = load_model('models/pre-trained/simple_CNN.81-0.96.hdf5', compile=False)
        self.model = load_model('models/gender_mini_XCEPTION.02-0.84.hdf5', compile=False)


        # self.model = load_model('C:/Users/Korisnik/Desktop/SoftComputing-master/soft-project/model3/gender_detection.model', compile=False)

        self.gender = ["woman", "man"]
        self.offsets = (20,20)

    def face_coordinates_offset(self, coordinates):
        x, y, w, h = coordinates
        x_off, y_off = self.offsets
        new_coordinates = (x-x_off, x + w + x_off, y - y_off, y + h + y_off)
        return new_coordinates

    def classify_gender(self, image, boxes):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if len(boxes) > 0:
            labels = []
            confidences =[]
            xs = []
            ys = []
            for i in range(0, len(boxes)):
                (fX, fY, fW, fH) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
                xs.append(fX)
                ys.append(fY)

                x, x2, y, y2 = self.face_coordinates_offset(boxes[i])

                # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
                roi = gray[round(y):round(y2), round(x):round(x2)]

                # roi = np.copy(image[round(y):round(y2), round(x):round(x2)])
                # print(roi)
                if len(roi) == 0: roi = gray[round(fY):round(fY + fH), round(fX):round(fX + fW)]
                try:
                    roi = cv2.resize(roi, (64, 64))
                except cv2.error:
                    continue

                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = self.model.predict(roi)
                confidence = preds.max()*100
                confidences.append(round(confidence))
                label = self.gender[preds.argmax()]
                labels.append(label)

            return labels, confidences
        else:
            return [], []

