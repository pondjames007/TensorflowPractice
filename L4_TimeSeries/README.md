# Lesson 4: Time Series

## Fixed Partitioning
* Split time series into 3 parts: **Training**, **Validation**, **Test**

## Evaluate Performance
### Metrics
* errors = forecast - real
* mse = np.square(errors).mean()
    * if large errors are potentially more dangerous and they cost you much more than small errors
* rmse = np.sqrt(mse)
* mae = np.abs(errors).mean()
    * mean absolute error
    * This does not penalize large errors as much as *mse* does
    * if your gain/loss is just proportional to the size of the error
* mape = np.abs(errors / x_valid).mean()
    * mean absolute percentage error


## Prepare Features
* Features: A number of values in the **series**
    * It is also called **window size**
* Label: The next value of the feature

### Make Dataset / Windowing Data
* **tf.data.Dataset**
```
    dataset = tf.data.Dataset(range(10))    # 0 ~ 9
    dataset = dataset.window(5, shift=1)    # size, shift
            # [
            #   [0,1,2,3,4],
            #   [1,2,3,4,5],
            #   [2,3,4,5,6],
            #   [3,4,5,6,7],
            #   [4,5,6,7,8],
            #   [5,6,7,8,9],
            #   [6,7,8,9],
            #   [7,8,9],
            #   [8,9],
            #   [9],
            # ]

    dataset = dataset.window(5, shift=1, drop_reminder=True)
            # [
            #   [0,1,2,3,4],
            #   [1,2,3,4,5],
            #   [2,3,4,5,6],
            #   [3,4,5,6,7],
            #   [4,5,6,7,8],
            #   [5,6,7,8,9],
            # ]

```
* Convert the dataset into **numpy** array
* Take the last element as **label**
* You can grab random number of data by **dataset.batch(num).prefetch(1)**


## Single Layer Neural Network
```
    window_size = 20
    batch_size = 32
    shuffle_buffer_size = 1000

    dataset = windowed_dataset(series, window_size, batch_size, shuffle_buffer_size)
    l0 = tf.keras.layers.Dense(1, input_size=[window_size])
    model = tf.keras.models.Sequential([l0])

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9), loss='mse')
    model.fit(dataset, epoch=100, verbose=0)
```

## Learning Rate Scheduler
* The learning rate can be changed in each epoch
* **tf.keras.callbacks**
* it should be called in **model.fit(callbacks=[])**
    * the same as the stop callback when reaching certain accuracy
```
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch/20))

    history = model.fit(dataset, epoch=100, callbacks=[lr_schedule])
```
* we can see the result and find out learning rate that make loss stable


## Apply RNN on Time Series
* With a RNN, you can feed in batches of sequences, and it will output a batch of forecasts
* One difference is that the full input shape when using RNNs is **3D**
    * shape = [batch_size, #time_steps, #dims]
* Ex: 
    * window_size = 30 timestamps
    * batching them in size of 4
    * shape = 4 * 30 * 1
    * at each timestamp, the **input = 4*1**
    * and the output from the memory cell **outputY = batch_size * unit_num**

### Sequence to Vector RNN
You want to get a single vector for each instance in the batch instead of the output
* Just ignore all outputs except the last one
* If you want to output a sequence, specify **return_sequence=True**
    * to stack one RNN on top of another
    * It is *DEFAULT*
```
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequence=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(1)
    ])
```
* input_shape = [batch_size, time_stamps]
* If using **return_sequence=True** on all layers
    * The Dense layer will get all sequences as its input
    * The Dense layer will be used at each timestamp
    * Become **Sequence to Sequence RNN**

### Lambda Layer
Perform arbitrary operations to effectively expand the functionality of Keras
```
    model = keras.models.Sequential([
        keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        keras.layers.SimpleRNN(20, return_sequence=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20),
        keras.layers.Dense(1),
        keras.layers.Lambda(lambda x: x * 100.0)
    ])
```

### Huber Loss
[Huber loss - Wikipedia](https://en.wikipedia.org/wiki/Huber_loss)


## Use LSTM
```
    tf.keras.backend.clear_session()    # clear internal vars
    dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x*100.0)
    ])
```


## Use Convolution
```
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(
            filter=32, kernel_size=5, strides=1, padding='causal,
            activation='relu', input_shape=[None, 1]
        ),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x*200.0)
    ])
```
* Since we remove the lambda function which reshapes the input, we need to specify the input_shape in Conv1D layer
* And the helper function should also update too