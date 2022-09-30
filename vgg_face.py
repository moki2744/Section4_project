import numpy as np
import os
import pickle
from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Convolution2D
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras_vggface.utils import preprocess_input, decode_predictions

TRAINING_DIR = "/Users/mok/Section4_Project/face_detect_train"
VALIDATION_DIR = "/Users/mok/Section4_Project/face_detect_test"
nb_class = 65
hidden_dim = 512
textFileNames = os.listdir(TRAINING_DIR)
batch_size = 8

def add_random_noise(x):
    x = x + np.random.normal(size=x.shape) * np.random.uniform(1,5)
    x = x - x.min()
    x = x / x.max()

    return x * 255.0

def make_dataset():

    training_datagen = ImageDataGenerator(
                        rescale=1./255,
                        rotation_range=90,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        brightness_range=(0.5, 1.3),
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='nearest',
                        preprocessing_function=add_random_noise)

    validation_datagen = ImageDataGenerator(
                        rescale=1./255)

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size=batch_size,
        target_size=(224, 224),
        class_mode = 'sparse',
        shuffle=True)

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        batch_size=batch_size,
        target_size=(224, 224),
        class_mode = 'sparse',
        shuffle=True)
    return train_generator, validation_generator

def make_model():
    vgg_notop = VGGFace(include_top=False, input_shape=(224, 224, 3))
    for layer in vgg_notop.layers:
        layer.trainable = False

    last_layer = vgg_notop.get_layer('pool5').output

    x = Flatten()(last_layer)
    x = Dense(hidden_dim, activation='relu')(x)
    x = Dense(hidden_dim, activation='relu')(x)
    out = Dense(nb_class, activation='softmax')(x)

    custom_vgg_model = Model(vgg_notop.input, out)
    custom_vgg_model.compile(loss='sparse_categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
    return custom_vgg_model

def model_train(model, train_generator, test_generator):
    callbacks = [EarlyStopping(patience=3, monitor='val_loss')]
    steps_per_epoch =  train_generator.n // batch_size
    validation_steps =  test_generator.n // batch_size
    nepochs = 30
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nepochs,
        validation_data=test_generator,
        validation_steps=validation_steps,
        callbacks = callbacks)

# def make_pickle():
#     with open('../model.pkl','wb') as pickle_file:
#     pickle.dump(boosting, pickle_file)

#     with open('../encoder.pkl','wb') as pickle_file:
#     pickle.dump(encoder, pickle_file)

train_generator, validation_generator = make_dataset()
# model = make_model()
# model_train(model, train_generator, validation_generator)
# model.save('vgg_face_trained_model.h5')

new_model = tf.keras.models.load_model('vgg_face_trained_model.h5')
test_loss, test_acc = new_model.evaluate(validation_generator, verbose=2)
print(test_loss, test_acc)
print(train_generator.class_indices)