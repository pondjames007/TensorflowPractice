"""
Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- 
i.e. you should stop training once you reach that level of accuracy.

Some notes:
    * It should succeed in less than 10 epochs, so it is okay to change epochs= to 10, but nothing larger
    * When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
    * If you add any additional variables, make sure you use the same names as the ones used in the class
"""

import tensorflow as tf

def train_mnist():
    # write callback
    class mycallbacks(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if log.set('acc') >= 0.99:
                print('\nReached 99% of accuracy so cancelling training!')
                self.model.stop_training = True
    
    callbacks = mycallbacks()
    ###

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize data
    x_train = x_train / 255
    x_test = x_test / 255
    ###

    model = tf.keras.models.Sequential([
        # add layers
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
        ###
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        # apply model fitting and add callback
        x_train, y_train, epochs=10, callbacks=[callbacks]
        ###
    )

    return history.epoch, history.history['acc'][-1]



train_mnist()