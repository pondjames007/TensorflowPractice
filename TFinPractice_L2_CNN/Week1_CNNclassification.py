"""
Use CNN to do Classification
"""

import subprocess

subprocess.check_output(['wget', '--no-check-certificate', 
'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip', 
'-O', './cats_and_dogs_filtered.zip'])

import os
import zipfile
import tensorflow as tf
from tensorflow._api.v1.keras.optimizers import RMSprop
from tensorflow._api.v1.keras.preprocessing.image import ImageDataGenerator


# The same as usual
local_zip = './cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./')
zip_ref.close()

base_dir = './cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['acc'])
##########################################

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale = 1/255)

test_datagen = ImageDataGenerator(rescale= 1/255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,              # This is the source directory for training images
    target_size= (150,150), # All images will be resized to 150x150
    batch_size= 20,         
    class_mode= 'binary'    # Since we use binary_crossentropy loss, we need binary labels
)

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size= (150,150),
    batch_size= 20,
    class_mode= 'binary'
)


history = model.fit_generator(
    train_generator,
    steps_per_epoch= 100,                   # 2000 images = batch_size * steps
    epochs= 100,
    validation_data= validation_generator,
    validation_steps= 50,                   # 1000 images = batch_size * steps
    verbose= 2
)
