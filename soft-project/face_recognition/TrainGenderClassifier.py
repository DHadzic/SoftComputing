from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from scipy.io import  loadmat
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator
from face_recognition.modelEmotionClassifier import mini_XCEPTION
from face_recognition.data_augmetation import ImageGenerator
import numpy as np


# parameters
batch_size = 32
num_epochs = 2
validation_split = .2
do_random_crop = False
patience = 100
num_classes = 2
dataset_name = 'wiki'
input_shape = (64, 64, 1)
if input_shape[2] == 1:
    grayscale = True
images_path = 'datasets/wiki_crop/'
#log_file_path = '../trained_models/gender_models/gender_training.log'
trained_models_path = 'pre-trained/gender_mini_XCEPTION'

def _load_imdb():
    face_score_treshold = 3
    dataset = loadmat('datasets/wiki_crop/wiki.mat')
    image_names_array = dataset['wiki']['full_path'][0, 0][0]
    gender_classes = dataset['wiki']['gender'][0, 0][0]
    face_score = dataset['wiki']['face_score'][0, 0][0]
    second_face_score = dataset['wiki']['second_face_score'][0, 0][0]
    face_score_mask = face_score > face_score_treshold
    second_face_score_mask = np.isnan(second_face_score)
    unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
    mask = np.logical_and(face_score_mask, second_face_score_mask)
    mask = np.logical_and(mask, unknown_gender_mask)
    image_names_array = image_names_array[mask]
    gender_classes = gender_classes[mask].tolist()
    image_names = []
    for image_name_arg in range(image_names_array.shape[0]):
        image_name = image_names_array[image_name_arg][0]
        image_names.append(image_name)
    return dict(zip(image_names, gender_classes))


def split_imdb_data(ground_truth_data, validation_split=.2, do_shuffle=False):
    ground_truth_keys = sorted(ground_truth_data.keys())
    if do_shuffle is not False:
        shuffle(ground_truth_keys)
    training_split = 1 - validation_split
    num_train = int(training_split * len(ground_truth_keys))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys

# data loader
ground_truth_data = _load_imdb()
train_keys, val_keys = split_imdb_data(ground_truth_data, validation_split)
print('Number of training samples:', len(train_keys))
print('Number of validation samples:', len(val_keys))


image_generator = ImageGenerator(ground_truth_data, batch_size,
                                 input_shape[:2],
                                 train_keys, val_keys, None,
                                 path_prefix=images_path,
                                 vertical_flip_probability=0,
                                 grayscale=grayscale,
                                 do_random_crop=do_random_crop)

# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# model callbacks
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/2), verbose=1)
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False)
callbacks = [model_checkpoint, early_stop, reduce_lr]

# training model
model.fit_generator(image_generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_keys) / batch_size),
                    epochs=num_epochs, verbose=1,
                    callbacks=callbacks,
                    validation_data=image_generator.flow('val'),
                    validation_steps=int(len(val_keys) / batch_size))





