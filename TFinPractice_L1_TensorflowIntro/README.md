# Lesson 1: Intro to Tensorflow

## Prerequisite
```
    import tensorflow as tf
    import numpy as np
    from tensorflow import keras

    import pyplotlib.pyplot as plt
```

## Make a model
* Sequential: It defines a **Sequence** of layers in the neural network
* Flatten: It flatten the layer into a **1 dimensional** set
* Dense: Add a layer of **full connected** neurons
```
    model = keras.models.Sequentials([layer1, layer2, ...])
    layer = keras.layers.Flatten()
    layer = keras.layers.Dense(units=, input_shape=, actication=)
```

### Activation Functions
* Relu: return X if X > 0 else 0
    * only pass values 0 or greater to the next layer
* Softmax: Pick the largest value
    *  [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05] -> [0, 0, 0, 0, 1, 0, 0, 0, 0] 
* Sigmoid: Good at doing **binary classification**, and the output only has 1 value
```
    activation = tf.nn.relu
    activation = tf.nn.softmax
    activation = tf.nn.sigmoid
```

## "Compile the Model"
* **Metric** will give you the info you ordered during training
* The case following will add '*accuracy*' metric
```
    model.compile(optimizer=, loss=, metrics=)

    optimizer = 'sgd' # stochastic gradient descent
    optimizer = tf.train.AdamOptimizer() # or 'adam'

    from keras.optimizers import RMSprop
    optimizer = RMSprop(lr= ) # can add learning rate


    loss = 'mean_squared_error'
    loss = 'sparse_categorical_crossentropy'
    loss = 'binary_crossentropy'


    metrics = ['accuracy']
```

## Train the model and add callbacks
* You can do callbacks during training, so that you may stop the training process earlier with some condition.
* **on_epoch_end**: this function will be called when an epoch ends.
```
    class myCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('loss') < 0.4):
                print("\nReached 60% accuracy so cancelling training!")
            
            self.model.stop_training = True
```
```
    model.fit(training_data_input, training_data_output, epochs=, callbacks=[]])
    callbacks = myCallback()
    # callbacks=[callbacks]
```

## Predict the Data / Evaluate the Data
```
    model.predict(input_data)  # get the predict value
    model.evaluate(test_data_input, test_data_output)  # get [loss, accuracy] of the test_data
```

* When you see the loss stops decteasing:
    * Overfitting -> try reduce training epochs, layers, ...etc.

## CNN
* Add Convolution and Pooling Layer
* For the **First Convolution**: It expects **a single tensor** containing everything, so instead of having 60000 *(28, 28, 1)* shape items in a list, we have **a single 4D list** that has shape **(60000, 28, 28, 1)**, and the same for the test images.
* NoOfFilters is not random, it is proven to be a good number
    * there are NoOfFilters Convolution Cores
* **MaxPooling2D(xsize, ysize)** -> pooling the **Max** value from a (xsize, ysize) values
```
    model = keras.models.Sequential([
        keras.layers.Conv2D(NoOfFilters, FilterSize(tuple), activation=, input_shape=[]),
        keras.layers.MaxPooling2D(xsize, ysize),
        keras.layers.Flatten(),
        keras.layers.Dense(units=, input_shape=, activation= ),
        keras.layers.Dense(unitz=, input_shape=, activation= )
    ])
```

* **model.summary()** allows you to inspect the model structure
* **output_shape** is important:
    * the original input is (28, 28, 1)
    * for the first convolution layer, it will be (None, 26, 26, 64) since convolution cannot go through the border pixel
```
    model.summary()
```

## ImageGenerator
* Use Tensorflow API to help automatically generate labels on your images as long as you put them in a correct folder hierarchy.
*   Images -> Training   -> Human -> yyy.jpg
	    			     -> Horse -> xxx.jpg
	       -> Validating -> Human -> zzz.jpg
				         -> Horse -> aaa.jpg
    * It will automatically label jpg files into their parent **folder name**

* from **tensorflow.keras.preprocessing.image** import **ImageDataGenerator**
* train_dir should be *Training* not Human or Horse
```
    # normalize the data
    train_datagen = ImageDataGenerator(rescale = 1/255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (300,300),
        batch_size = 128,
        class_mode = 'binary'
    )
```

## Train the Model with Generator
* steps_per_epoch:
    * we have 1024 training images, and our batch_size in train_generator = 128
    * 1024/128 = 8
* validation_steps:
    * we have 256 images for validation, batch_size = 32
    * 256/32 = 8
* verbose:
    * sepcifies how much to display while training
    * with verbose set to 2, we'll get a little less animation hiding in the epoch progress

```
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 8,
        epochs = 15, 
        validation_data = validation_generator,
        validation_steps = 8,
        verbose = 2
    )
```