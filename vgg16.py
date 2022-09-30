import tensorflow as tf
import keras_vggface
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# set my dataset directory
TRAINING_DIR = "/Users/mok/Section4_Project/face_detect_train"
VALIDATION_DIR = "/Users/mok/Section4_Project/face_detect_test"
batch_size = 8

# Random Noise Function
def add_random_noise(x):
    x = x + np.random.normal(size=x.shape) * np.random.uniform(1,5)
    x = x - x.min()
    x = x / x.max()

    return x * 255.0

def make_dataset():
    training_datagen = ImageDataGenerator(
                        rescale=1./255,
                        rotation_range=40,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        brightness_range=(0.5, 1.3),
                        horizontal_flip=True,
                        fill_mode='nearest',
                        preprocessing_function=add_random_noise)

    validation_datagen = ImageDataGenerator(
                        rescale=1./255)

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size=batch_size,
        target_size=(224, 224),
        class_mode = 'categorical',
        shuffle=True)

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        batch_size=batch_size,
        target_size=(224, 224),
        class_mode = 'categorical',
        shuffle=True)
    return train_generator, validation_generator

def show_dataset(dataset):
    img, label = next(dataset)
    plt.figure(figsize=(20,20))

    for i in range(8):
        plt.subplot(3,3,i+1)
        plt.imshow(img[i])
        # plt.title(label[i])
        plt.axis('off')
    plt.show()

def make_model():
    base_model = tf.keras.applications.VGG16(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base_model.trainable = False

    out_layer = tf.keras.layers.Conv2D(128, (1,1), padding='same', activation=None)(base_model.output)
    out_layer = tf.keras.layers.BatchNormalization()(out_layer)
    out_layer = tf.keras.layers.ReLU()(out_layer)

    out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer)
    out_layer = tf.keras.layers.Dense(65, activation='softmax')(out_layer)

    # Make New model
    model = tf.keras.models.Model(base_model.input, out_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

train_generator, validation_generator = make_dataset() #데이터셋 생성
show_dataset(train_generator)
# model = make_model() #모델 생성
# model.fit(train_generator, epochs=25, validation_data=validation_generator, verbose=1)
# model.save("saved_model.h5")