import os
import zipfile
import random
import tensorflow as tf
from tensorflow._api.v1.keras.optimizers import RMSprop
from tensorflow._api.v1.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

# Everyhing else is the same 

# Do Image Augmentation
TRAINING_DIR = '/tmp/cats-v-dogs/training' 
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True,
    fill_mode = 'nearest'
) 

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,              # This is the source directory for training images
    target_size= (150,150), # All images will be resized to 150x150
    batch_size= 20,         
    class_mode= 'binary'    # Since we use binary_crossentropy loss, we need binary labels
)

VALIDATION_DIR = '/tmp/cats-v-dogs/testing'
validation_datagen = ImageDataGenerator(rescale=1/255) 
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    batch_size=128,
    class_mode='binary'

)


# Everyhing else is the same 