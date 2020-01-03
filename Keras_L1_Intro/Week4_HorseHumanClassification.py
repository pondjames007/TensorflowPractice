"""
Real Case Application: Horse Human Classification
"""

# Download Images
import subprocess

subprocess.check_output(['wget', '--no-check-certificate', 
'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip', 
'-O', './horse-or-human.zip'])

# Load files
import os
import zipfile

local_zip = './horse-or-human.zip'
zip_ref = zipfile.Zipfile(local_zip, 'r')
zip_ref.extractall('./horse-or-human')
zip_ref.close()

# Define Directories
train_horse_dir = os.path.join('./horse-or-human/horses')
train_human_dir = os.path.join('./horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)

print('total training horse images: ', len(os.listdir(train_horse_dir)))
print('total training human images: ', len(os.lsitdir(train_human_dir)))

# Draw Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

## parameters for the graph
nrows = 4
ncols = 4

## index iterating over images
pic_index = 0

## set up matplotlib fig, and size it to fit 4*4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_index-8: pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) for fname in train_human_names[pic_index-8: pic_index]]

for i, img_path in enumerate(next_horse_pix + next_human_pix):
    ## set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i+1)
    sp.axis('Off') ## don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()


# Build Model
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.summary()

# using the RMSprop optimization algorithm is preferable to stochastic gradient descent (SGD), 
# because RMSprop automates learning-rate tuning for us. 
# (Other optimizers, such as Adam and Adagrad, also automatically adapt the learning rate during training, 
# and would work equally well here.)
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])


# Data Preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1/255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    './horse-or-human', # Source Dir for training images
    target_size = (300, 300), # All images will be resized to 300x300
    batch_size = 128,
    class_mode = 'binary' # since we use binary_crossentropy loss, we need binary labels
)




# Training
history = model.fit_generator(
    train_generator,
    steps_per_epoch= 8, 
    epochs= 15,
    verbose= 1
)

