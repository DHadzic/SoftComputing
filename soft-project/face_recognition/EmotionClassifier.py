import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2


class EmotionClassifier:
    def __init__(self):

        # self.model = load_model('models/_mini_XCEPTION.01-0.43.hdf5', compile=False)
        self.model = load_model('models/_mini_XCEPTION.102-0.66.hdf5', compile=False)

        self.emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

    def classify_emotion(self,image,boxes):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if len(boxes) > 0:
            labels = []
            xs = []
            ys = []
            for i in range(0, len(boxes)):
                (fX, fY, fW, fH) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
                xs.append(fX)
                ys.append(fY)
                # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
                roi = gray[round(fY):round(fY + fH), round(fX):round(fX + fW)]
                try:
                    roi = cv2.resize(roi, (64, 64))
                except cv2.error:
                    continue

                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                # print(roi.shape)
                preds = self.model.predict(roi)[0]
                # emotion_probability = np.max(preds)
                label = self.emotions[preds.argmax()]
                labels.append(label)

            return self.add_emotion_text(image,labels,xs,ys)

        else:
            return image

    def add_emotion_text(self,image,labels,xs,ys):

        for i in range(0, len(labels)):
            cv2.putText(image,labels[i],(int(xs[i]), int(ys[i]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        return image
