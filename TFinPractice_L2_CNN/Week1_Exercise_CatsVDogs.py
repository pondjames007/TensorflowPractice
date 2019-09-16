import os
import zipfile
import random
import tensorflow as tf
from tensorflow._api.v1.keras.optimizers import RMSprop
from tensorflow._api.v1.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

import subprocess

subprocess.check_output(['wget', '--no-check-certificate', 
'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip', 
'-O', './cats_and_dogs_filtered.zip'])

# Create Directories for the dataset
try:
    os.mkdir('./cats-v-dogs')
    os.mkdir('./cats-v-dogs/training')
    os.mkdir('./cats-v-dogs/testing')
    os.mkdir('./cats-v-dogs/training/cats')
    os.mkdir('./cats-v-dogs/training/dogs')
    os.mkdir('./cats-v-dogs/testing/cats')
    os.mkdir('./cats-v-dogs/testing/dogs')
except OSError:
    pass


# Split data into 9:1 of training:testing and 
# put them into directories respectively
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = os.listdir(SOURCE)
    files = random.sample(files, len(files))
    num = int(SPLIT_SIZE * len(files))

    for i, _file in enumerate(files):
        if i <= num:
            dst = TRAINING
        else:
            dst = TESTING

        if not os.path.getsize(SOURCE + _file): continue

        copyfile(SOURCE + _file, dst + _file)
###

CAT_SOURCE_DIR = './PetImages/Cat'
TRAINING_CATS_DIR = "./cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "./cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "./PetImages/Dog/"
TRAINING_DOGS_DIR = "./cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "./cats-v-dogs/testing/dogs/"

split_size = 0.9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)


# Define a Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])


# Use Image Generator
TRAINING_DIR = './cats-v-dogs/training'
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    batch_size=128,
    class_mode='binary'
)

VALIDATION_DIR = './cats-v-dogs/testing'
validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150, 150),
    batch_size=128,
    class_mode='binary'
)


history = model.fit_generator(
    train_generator,
    epochs=15, 
    verbose=1,
    validation_data=validation_generator
)