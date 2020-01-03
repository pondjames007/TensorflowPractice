"""
Exercise 3
 
Improve MNIST to 99.8% accuracy or more using only a single convolutional layer and a single MaxPooling 2D. 
You should stop training once the accuracy goes above this amount. 
It should happen in less than 20 epochs, so it's ok to hard code the number of epochs for training, 
but your training must end once it hits the above metric. If it doesn't, then you'll need to redesign your layers.

When 99.8% accuracy has been hit, you should print out the string "Reached 99.8% accuracy so cancelling training!"
"""

import tensorflow as tf

def train_mnist_conv():
    # add callback
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.set('acc') >= 0.998:
                print('\nReach 99.8% accuracy so cancelling training!')
                self.model.stop_training = True
    
    callbacks = myCallback()
    ###

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    # Reshape data to make it fit Convolution input
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255
    ###

    model = tf.keras.models.Sequential([
        # Design Layers
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=[28, 28, 1]),
        tf.keras.layers.MaxPooling2D(4,4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        ###
    ])


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    # model fitting
    history = model.fit(training_images, traing_labels, epoch=20, callbacks=[callbacks])
    ###

    return history.epoch, history.history['acc'][-1]


    train_mnist_conv()