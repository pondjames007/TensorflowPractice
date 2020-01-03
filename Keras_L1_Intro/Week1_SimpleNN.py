import tensorflow as tf
import numpy as np
from tensorflow import keras

"""
the simplest possible neural network. It has 1 layer, and that has 1 neuron, and the input shape to it is just 1 value.
* the parameter put in Sequential() should be a list [keras.layers.Dense(units=, input_shape=[1])]
* because each element in the list is a layer
"""

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

"""
Now compile our Neural Network. 
When we do so, we have to specify 2 functions, a loss and an optimizer.
"""

model.compile(optimizer='sgd', loss='mean_squared_error')

# Providing the Data
x = np.array([-1, 0, 1, 2, 3, 4], dtype='float')
y = np.array([-3,-1, 1, 3, 5, 7], dtype='float')

# Training
model.fit(x, y, epochs=500)

# Predict
model.predict([10])