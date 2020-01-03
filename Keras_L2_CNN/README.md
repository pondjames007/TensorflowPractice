# Lesson 2: Convolutional Neural Network

## Image Augmentation
Apply preprocessing on the images you have to get more variations on the dataset
* from **keras.preprocessing** import **ImageDataGenerator**
```
    train_datagen = ImageDataGenerator(
        rescale = 1/255,
        rotation_range = 40,        # [0, 180], randomly rotate some degree
        width_shift_range = 0.2,    # [0, 1], shift the image in proportion
        height_shift_range = 0.2,   
        shear_range = 0.2,          # shear the image in proportion
        zoom_range = 0.2,           # zoom the image in proportion
        horizontal_flip = True,     # flip the image
        fill_mode = 'nearest'       # fill the lost pixels after transformation with certain mode
    )
```

* However, if the validation dataset does not have the same randomness, it may result in fluctuations.
* We don't just need a broad set of images for training but also need them for testing to make image augmentation help.


## Transfer Learning
* Take an **existing model** that has trained on **far more data**, and **use the features** the model learned.
* Freeze/Lock the layers you want and train those are not locked
* use functions in **tf.keras.layers** to see the structure of a layer

### Inception V3
* A pre-trained model using **ImageNet** dataset
```
    from tf.keras.applications.inceptions_v3 import InceptionV3

    local_weights_file = '....'

    pretrained_model = InceptionV3(
        input_shape = (150, 150, 3),
        
        include_top = False,    # InceptionV3 has a fully connected layer at the top,
                                # setting to False will ignore the top layer and
                                # go straight to the convolutional layer.
        
        weights = None          # don't use the built-in weights
    )
```

* During instantiation: Use built-in weight or not; Include the top layer or not
* After instantiation: Lock the layers you want

### Get a Layer
* layer_you_want = pretrained_model.get_layer('xxx')
* last_output = layer_you_want.output -> set the layer you take as the output

### Connect to our own model from the layer we set to be the output
```
    from tf.keras import Model

    last_layer = pretrained_model.get_layer('mixed7')
    last_output = last_layer.output


    # Another way to set up a model

    # Flatten the input, which is *last_output*
    x = tf.keras.layers.Flatten()(last_output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Create a model that passing at the input and the layers you define
    model = Model(pretrained_model.input, x)
    model.compile(....)
```

## Dropout
* **Remove** a random number of neurons
* Neighbor neurons always end up with similar weights, which can lead to **Overfitting**
* A neuron can **Over-weigh** the input from a neuron in the **previous layer**, and can over **specialize** as a result
```
    # Add a dropout when defining layers of your model
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    # Add here
    x = tf.keras.layers.Dropout(0.2)(x) # [0, 1], proportion to dropout
    ###
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
```


## Multiple Categories Classifier
* Change **class_mode** into **categorical** in ImageDataGenerator
* Change the last layer of your model to **Dense(x, activation='softmax')**
* Change **loss** to **sparse_categorical_crossentropy** in model.compile()