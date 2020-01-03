"""
A Computer Vision Example
using fashion mnist
"""

import numpy as np
from tensorflow import keras
import tensorflow as tf

# Get fashion Mnist
fashion_mnist = keras.datasets.fashion_mnist

# the dataset will return 2 tuples (training and test) when executing load_data()
(training_data, training_labels), (test_data, test_labels) =  fashion_mnist.load_data()

import matplotlib.pyplot as plt

plt.imshow(training_data[0])


# normalize the data to get better efficiency
# the data is read only -> cannot use /=
training_data = training_data / 255
test_data = test_data / 255

"""
Sequential: That defines a SEQUENCE of layers in the neural network
Flatten: Takes the square image and turn it into a 1 dimensional set
Dense: Adds a layer of neurons
    * Each layer of neurons need an activation function to tell them what to do.
        * Relu: return x if x > 0 else 0
        * Softmax: make the largest value to 1, else 0
"""

model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_data, training_labels, epochs=5)

model.evaluate(test_data, test_labels)



"""
Exercise 1
"""

calssifications = model.predict(test_data)

# print out the prediction of test_data[0] in probability for 10 categories
print(classifications[0])
# print out the category of test_data[0] to verify the prediction
print(test_labels[0])


"""
Exercise 2
"""

# increase the neuron number from 512 to 1024 for the first layer
# it takes longer time but the result is more accurate
model2 = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model2.fit(training_data, training_labels, epochs=5)

model2.evaluate(test_data, test_labels)

"""
Exercise 3
What would happen if you remove the Flatten() layer. Why do you think that's the case?
"""
# You get an error about the shape of the data. 
# it reinforces the rule of thumb that the first layer in your network should be the same shape as your data. 
# Right now our data is 28x28 images, and 28 layers of 28 neurons would be infeasible, 
# so it makes more sense to 'flatten' that 28,28 into a 784x1. 
# Instead of wriitng all the code to handle that ourselves, we add the Flatten() layer at the begining, 
# and when the arrays are loaded into the model later, they'll automatically be flattened for us.

"""
Exercise 4
Consider the final (output) layers. Why are there 10 of them? 
What would happen if you had a different amount than 10? For example, try training the network with 5
"""
# You get an error as soon as it finds an unexpected value. 
# Another rule of thumb -- the number of neurons in the last layer should match the number of classes you are classifying for. 
# In this case it's the digits 0-9, so there are 10 of them, hence you should have 10 neurons in your final layer.


"""
Exercise 5
Consider the effects of additional layers in the network. 
What will happen if you add another layer between the one with 512 and the final layer with 10.
"""
# Ans: There isn't a significant impact -- because this is relatively simple data. 
# For far more complex data (including color images to be classified as flowers that you'll see in the next lesson), 
# extra layers are often necessary.

"""
Exercise 6
Consider the impact of training for more or less epochs. Why do you think that would be the case?
"""
# Try 15 epochs -- you'll probably get a model with a much better loss than the one with 5 Try 30 epochs -- 
# you might see the loss value stops decreasing, and sometimes increases. 
# This is a side effect of something called 'overfitting' which you can learn about [somewhere] 
# and it's something you need to keep an eye out for when training neural networks. 
# There's no point in wasting your time training if you aren't improving your loss, right! :)


"""
Add callback to stop training when meets the criteria
"""
class myCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
model2.fit(training_data, training_labels, epochs=15, callbacks=[callbacks])

