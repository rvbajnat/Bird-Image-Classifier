import tensorflow as tf 
import keras
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import adam
import matplotlib.pyplot as plt
import os
from keras.models import load_model

def load_data():
    # Generator for training data
    train_generator = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)
    train_data = train_generator.flow_from_directory('birdDataSet/train', target_size=(224,224), batch_size = 16, class_mode='categorical')

    # Generator for validation data
    valid_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    valid_data = valid_generator.flow_from_directory('birdDataSet/valid', target_size=(224,224), class_mode='categorical', batch_size=16)

    # Generator for test data
    test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_data = test_generator.flow_from_directory('birdDataSet/test', target_size=(224,224), class_mode='categorical', batch_size=16)
    return train_data, valid_data, test_data

def define_model():
    # setting base model with pre-trained weights
    baseline_model = keras.applications.VGG16 (include_top=False, weights='imagenet', input_shape=(224,224,3))
    # freezing model
    baseline_model.trainable = False
    # creating new predicting layers
    model = Sequential()
    model.add(baseline_model)
    model.add(Flatten())
    model.add(Dense(2048, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.4))
    model.add(Dense(2048, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.4))
    model.add(Dense(225, activation='softmax', kernel_initializer='glorot_normal'))
    # compile model
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def visualize(history):
    # plotting loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='validation_loss')
    plt.legend()
    plt.show()
    # plotting accuracy
    plt.plot(history.history['accuracy'], label='acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()

def test_model():
    model = define_model()
    train_data,valid_data,test_data = load_data()
    history = model.fit(train_data, epochs=30, validation_data=valid_data, workers=10)
    visualize(history)
    model.save("model.h5")

def evaluate_model():
    model = load_model('modelRefine.h5')
    train_data,valid_data,test_data = load_data()
    score = model.evaluate(test_data, workers=10)
    print(score)

def fineTune():
    model = load_model('model.h5')
    train_data,valid_data,test_data = load_data()
    base = model.layers[0]
    base.trainable = True
    set_trainable = False
    for layer in base.layers:
        if layer.name == 'block4_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data, epochs=30, validation_data=valid_data, workers=10)
    visualize(history)
    model.save("model.h5")

def refine():
    model = load_model('model.h5')
    train_data,valid_data,test_data = load_data()
    model.compile(optimizer=keras.optimizers.Adam(1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data, epochs=10, validation_data = valid_data, workers=10)
    visualize(history)
    model.save("modelRefine.h5")
# fineTune()
evaluate_model()
# refine()
# keras.backend.clear_session()
