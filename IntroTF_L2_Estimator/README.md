# Lesson 2: Estimator

## tf.estimator()
* High level Tensorflow APIs
* It wraps up all layers, losses, ...etc.
* You can build your own model from layers using tf API

### Pre-made Estimators
* They are in **tf.estimator.Estimator()**
    * LinearRegressor, DNNRegressor, DNNLinearCombinedRegressor, ...
    * LinearClassifier, DNNClassifier, **DNNLinearCombinedClassifier**, ...
* DNNLinearCombinedClassifier is widely used
```
    # Specify Features
    fearcols = [
        tf.feature_column.numeric_column("sq_footage"),
        tf.feature_column.categorical_with_vocabulary_list(
            "type", ["house", "apt"]
        )
    ]

    # Use Pre-made Estimators to build a model
    model = tf.estimator.LinearRegressor(featcols)
```
* Inputs to the estimator model are in the form of **feature columns**

## Checkpoints
* Continue Training
* Resume from Failure
* Predict from trained model

### Specify Checkpoint
* Specify a folder to put checkpoint data (Do the same specification when you want to restore from the checkpoint, TF will automatically check if there is checkpoint data in the folder or not)
```
    model = tf.estimator.LinearRegressor(featcols, './checkpoint')
```
* Remember to delete the folder when you want to retrain or change the model


## Put data into TF from Numpy (or Pandas)
```
    def numpy_train_input_fn(sqft, prop_type, price):
        return tf.estimator.inputs.numpy_input_fn(
            x = {"sq_footage": sqrt, "type": prop_type},
            y = price,
            batch_size = 128,
            num_epochs = 10,
            shuffle = True,
            queue_capacity = 1000
        )
```
* By default, the training process will stop when the data is exhausted or num_epochs times of data is exhausted
* Add some variants can do additional steps
```
    model.train(numpy_train_input_fn(...)) # train 10 epochs on all data
    
    # train "additional" 1000 steps from the checkpoint
	# one step = one batch of data
    model.train(numpy_train_input_fn(...), steps = 1000)
    
	# train "additional" steps from the last checkpoint if the step count is not 100 yet
    model.train(numpy_train_input_fn(...), max_steps=1000)	
```

LAB: *training-data-analyst > courses > machine_learning > deepdive > 03_tensorflow > labs*and open*b_estimator.ipynb*

## Load Data into Memory
* Use **tf.data.Dataset**
* It can help us create input functions for out model that load data progressively

### tf.data.Dataset
* .TextLineDataset -> for CSV, txt
* .TFRecordDataset -> for TF Record
* .FixedLengthRecordDataset -> for FixedLength

* It generates the input of TF model and connect to the data (in batches)

EX: Read one CSV file using TextLineDataset
```
    def decode_line(row):
        cols = tf.decode_csv(row, record_defaults=[[0], ['house'], [0]])
        features = {'sq_footage': cols[0], 'type': cols[1]}
        labels = cols[2]

        return features, cols

    dataset = tf.data.TextLineDataset("train_1.csv").map(decode_line)

    dataset = dataset.shuffle(1000).repeat(15).batch(128)


    def input_fn():
        features, cols = dataset.make_one_shot_iterator().get_next()
        
        return features, cols

    model.train(input_fn)
```
* **model.train()** launches the training loop
* the model recieves data from its input nodes
    * it is defined in **input_fn()**
    * it will automatically return one batch of data during the training loop
* the dataset **shuffles** the data, repeats 15 **epochs**, and has **batch** with size 128
* the dataset is read from the csv file by TextLineDataset and is transformed from text lines into dataset of features and label


### Load from multiple (shard) files
```
    dataset = tf.data.list_files("train_csv-*")
                     .flat_map(tf.data.TextLineDataset) # map all files into one dataset
                     .map(decode_line)
```
* **flat_map**: one to many transformation
    * when loading a file with text line dataset, one file name becomes a collection of text lines
* map: one to one transformation
    * parsing a line of text

LAB: *training-data-analyst > courses > machine_learning > deepdive > 03_tensorflow > labs*and open*c_dataset.ipynb*.


## Distribution
* Use **tf.estimator.train_and_evaluate()**
* Declare a variable estimator (ex: tf.estimator.LinearRegressor()) 
* pass the estimator into train_and_evaluate function

### Data Parallelism
* Replicate the model to multiple users
* Have a server to store all parameters
* Everything is wrapped in **tf.estimator.train_and_evaluate()**

The only thing you need to do:
1. choose an estimator
2. add run_config
3. provide train spec and evaluate spec
```
    estimator = tf.estimator.LinearRegressor(       <- 1.   
                    feature_columns = featcols,
                    config = run_config             <- 2.
    )

    tf.estimator.train_and_evaluate(
                    estimator,
                    train_spec,                     <- 3.
                    eval_spec                       <- 3.
    )

```

#### RunConfig
It tells the estimator **where** and **how often** to write **Checkpoints** and **Tensorboardlogs("summaries")**
```
    run_config = tf.estimator.RunConfig(
                    model_dir = '...',
                    save_summary_steps = 100,
                    save_checkpoints_steps = 2000
    )
```

#### TrainSpec
It tells the estimator **how to get** the training data
```
    train_spec = tf.estimator.TrainSpec(
                    input_fn = train_input_fn,      <- Use Datasets API
                    max_steps = 50000
    )
```

#### EvalSpec
It controls the evaluation and the checkpoints of the model since they happen at the same time
```
    eval_spec = tf.estimator.EvalSpec(
                    input_fn = eval_input_fn,
                    step = 100,                     <- evals on 100 batches
                    throttle_secs = 600,            <- eval no more than 10 min
                    exporters = ...
    )    
```
* It takes the latest checkpoint to do the evaluation
* You can not make the evaluation more frequently than make a checkpoint


### Shuffle Dataset
* Even the data itself has already shuffled on disk, in distributed training, every worker will load in the same data at the same time
* It is important to make sure every worker is getting different shuffled data
```
    tf.data.Dataset.list_files('...')
                   .shuffle(100)                    <- add this line
                   .flat_map(...)
                   .map(...)
```



## Tensorboard
**Tensorboard** always points to the output directory specified in **RunConfig**, and it is default at **localhost:6006**
* Pre-made Estimator exports relevant metrics, embeddings, histograms, ...etc
* You an see **Tensorflow Graph**
* If you are using custom Estimator model, you can add summaries for Tensorboard with a single line
    * tf.summary
        * .scalar, .image, .audio, .text, .histogram


## Deploy the model
### Exporter Parameter in EvalSpec
It is what defines a complete model
```
    export_latest = tf.estimator.LatestExporter(
                        serving_input_receiver_fn = serving_input_fn
    )

    eval_spec = ...
```

### Serving Input Function
* Same as training input function, it also runs **once**
* It instantiate a graph to parse input(JSON, REST API, ...) and do transformation to what the model wants
```
    def serving_input_fn():
        json = {
            'sq_footage': tf.placeholder(tf.int32, [None]#batch_size#),
            'prop_type': tf.placeholder(tf.int32, [None])
        }

        # ... transformation ...

        features = {
            'sq_footage': json['sq_footage'],
            'type': json['prop_type']
        }

        return tf.estimator.export.ServingInputReceiver(features, json)


    exp = tf.estimator.LatestExporter("pricing", serving_input_fn)

    eval_spec = tf.estimator.EvalSpec(exporter=exp, ...)
```
* The **exporter** will save a **checkpointed version of the model** along with the **transformation info** into an exported model file that is ready to be deployed
* Which checkpoint will be chosen?
    * depends on which exporter is used


### Save input function that decodes images
The model will expect the **decompressed images**
```
    def serving_input_fn():
        json = {'jpeg_bytes': tf.placeholder(tf.string, [None])}

        def decode(jpg):
            pixels = tf.image.decode_jpeg(jpeg, channels=3)

            return pixels

        pics = tf.map_fn(decode, json['jpeg_types'], dtype=tf.uint8)

        features = {'pics': pics}

        return tf.estimator.export.ServingInputReceiver(features, json)
```
* tf.string: a byte string format



## Conclusion
* Build models
* Pre-made estimators
* Custom estimators
* Datasets API to batch data
* Train&Eval to distribute training
* Tensorboard to monitor the model
* Exporter to deploy

LAB:*training-data-analyst > courses > machine_learning > deepdive > 03_tensorflow > labs*and open*d_traineval.ipynb*.