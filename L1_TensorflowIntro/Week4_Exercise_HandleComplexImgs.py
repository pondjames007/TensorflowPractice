"""
Exercise 4

Happy or sad dataset which contains 80 images, 40 happy and 40 sad. 
Create a convolutional neural network that trains to 100% accuracy on these images, 
which cancels training upon hitting training accuracy of >.999

Hint -- it will work best with 3 convolutional layers.
"""
import tensorflow as tf
import os
import zipfile
import urllib
urllib.request.urlretrieve("https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip",
                           filename="./happy-or-sad.zip")

zip_ref = zipfile.ZipFile("./happy-or-sad.zip", 'r')
zip_ref.extractall("./h-or-s")
zip_ref.close()


def train_happy_sad_model():
    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.set('acc') > DESIRED_ACCURACY:
                print('\nReached Desired Accuracy so cancelling')
                self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])



    from tensorflow.keras.optimizers import RMSprop

    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
        

    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1/255)

    train_generator = train_datagen.flow_from_directory(
        "./h-or-s/",
        target_size=(100,100),
        batch_size=10,
        class_mode='binary'
    
    )
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        verbose=1,
        callbacks=[callbacks]
    )
    # model fitting
    return history.history['acc'][-1]


train_happy_sad_model()

