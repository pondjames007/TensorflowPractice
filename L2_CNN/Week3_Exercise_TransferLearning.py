import os
from tensorflow._api.v1.keras import layers, Model
from tensorflow._api.v1.keras.applications.inception_v3 import InceptionV3

import subprocess

subprocess.check_output(['wget', '--no-check-certificate', 
'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', 
'-O', './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'])

local_weight_file = './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pretrained_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top = False,
    weights = None
)

pretrained_model.load_weights(local_weight_file)


for layer in pretrained_model.layers:
    layer.trainable = False


last_layer = pretrained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output



from tensorflow._api.v1.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1024 hidden units and Relu activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pretrained_model.input, x)
model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])