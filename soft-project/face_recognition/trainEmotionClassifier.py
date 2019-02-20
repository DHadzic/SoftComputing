from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import cv2
from face_recognition.modelEmotionClassifier import mini_XCEPTION


def load_fer2013(image_size=(48, 48)):
    data = pd.read_csv('fer2013.csv')
    pixels = data['pixels'].tolist()[0:1000]
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    emotions = emotions[0:1000]
    return faces, emotions


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data


if __name__ == "__main__":

    # parameters
    batch_size = 32
    num_epochs = 100
    input_shape = (64, 64, 1)
    validation_split = .2
    verbose = 1
    num_classes = 7
    patience = 50
    base_path = 'models/'

    # data generator
    data_generator = ImageDataGenerator(
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=.1,
                            horizontal_flip=True)

    # model parameters/compilation
    model = mini_XCEPTION(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()


    datasets = ['fer2013']
    for dataset_name in datasets:
        print('Training dataset:', dataset_name)

        # callbacks
        log_file_path = base_path + dataset_name + '_emotion_training.log'
        csv_logger = CSVLogger(log_file_path, append=False)
        early_stop = EarlyStopping('val_loss', patience=patience)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                      patience=int(patience/4), verbose=1)
        trained_models_path = base_path + dataset_name + '_mini_XCEPTION'
        model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
        model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                        save_best_only=True)
        callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

        # loading dataset
        faces, emotions = load_fer2013(input_shape[:2])
        faces = preprocess_input(faces)
        num_samples, num_classes = emotions.shape
        train_data, val_data = split_data(faces, emotions, validation_split)
        train_faces, train_emotions = train_data
        model.fit_generator(data_generator.flow(train_faces, train_emotions,
                                                batch_size),
                            steps_per_epoch=len(train_faces) / batch_size,
                            epochs=num_epochs, verbose=1, callbacks=callbacks,
                            validation_data=val_data)