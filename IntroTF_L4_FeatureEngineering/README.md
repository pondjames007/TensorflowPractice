# L4: Feature Engineering

## Raw data to features
* Turn raw data into the format you want

## Good Features
* Related to objective
* Known at prediction time
* Numeric with meaningfule magnitude
* Have enough examples
* Bring human insight to problem

## Representing Features
Features in Raw data should be converted into **feature columns**
* **Numeric values** can be used as
    * **tf.feature_column.numeric_column(*feature_name*)**
* **Categorical Variables** should be **one-hot encoded**
    * if there are 5 categories:
        * [00001, 00010, 00100, 01000, 10000] to represent 5 categories
    * **tf.feature_column.cstegorical_column_with_vocabulary_list(*feature_name*, Vocabulary_list=[*category elements*])**

Preprocess data to create vocabulary of keys
* The vocabulary and the mapping of the vocabulary needs to be identical at prediction time

### Options for encoding categorical data
* If you know the keys beforehand
```
    tf.feature_column.categorical_column_with_vocabulary_list(
        'employee_Id', vocabulary_list=[a, s, f, g, h]
    )
```
* If your data is already indexed; i.e., has integers [0, N):
```
    tf.feature_column.categorical_column_with_identity(
        'employee_Id', num_buckets=5
    )
```
* If you don't have a vocabulary of all possible values
```
    tf.feature_column.categorical_column_with_hash_bucket(
        'employee_Id', hash_bucket_size=500
    )
```

* tf.feature_column.bucketized_column
    * discretize floating points into smaller bucket categories

* Remember to add a column to represent missing data

LAB: *training-data-analyst > courses > machine_learning > deepdive > 04_features*and open*a_features.ipynb*.


## Preprocessing and Feature Creation
* In BigQuery or Apache Beam
    * Remove examples that you don't want to train on
    * Compute vocabularies for categorical columns
    * Compute aggregate statistics for numeric columns
* In Apache Beam only
    * Compute time-windowed statistics (e.g. number of products sold in previous hour) to be input features
* In Tensorflow or Apache Beam
    * Scaling, discretization, etc. of numeric features
    * Splitting, lower-casing, etc. of textual features
    * Resizing of input images
    * Normalizing volume level of input audio

You can use SQL statements in BigQuery to do Data Preprocessing
If you are using SQL to do preprocessing, you need to implement **exactly** the same preprocessing logics in Tensorflow

Using Tensorflow can access to helper APIs and the pipelines are the same in both training and serving.

Using Apache Beam can take the same code to preprocess features in training and evaluation and serving.

### Preprocessing Logic in Tensorflow
There are 2 ways to create features
1. Features are preprocessed in **input_fn(train, eval, serving)**
    * If the feature preprocessing step that you need **is not available in the Tensorflow APIs**, you can modify the functions used in the input parameters during training, validation, and test.
```
    features['capped_rooms'] = tf.clip_by_value(
        features['rooms'],
        clip_value_min = 0,
        clip_value_max = 4
    )
```

You need to implement your own preprocessing code
```
    def add_engineered(features):
        lat1 = features['pickuplat']
        ...
        dist = tf.sqrt(latdiff*latdiff + londiff*longdiff)
        features['euclidean'] = dist

        return features


    def input_fn():
        features = ...
        label = ...

        return add_engineered(features), label


    def serving_input_fn():
        feature_placeholder = ...
        features = ...

        return tf.estimator.export.ServingInputReceiver(
                    add_engineered(features), feature_placeholders
                )
```

2. Feature colummns are passed into the estimator during construction
    * If you want to change a real value feature into a discrete one
```
    lat = tf.feature_column.numeric_column('latitude')
    dlat = tf.feature_column.bucketized_column(
        lat,
        boundaries=np.arange(32, 42, 1).tolist()
    )
```

You can just call Tensorflow API functions
```
    def build_estimator(model_dir, nbuckets):
        latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
        b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
        b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)

        return tf.estimator.LinearRegressor(
                    model_dir = model_dir,
                    feature_columns = [..., b_plat, b_dlat, ...]
                )
```
* latbuckets -> define the range for the buckets (in nbuckets)


## Apache Beam / Cloud Dataflow
Beam is a way to write elastic data processing pipelines
* Pipeline: a sequence of steps that change data from one format into another

You can implement data processing pipeline and write the code using Apache Beam, then deploy to Dataflow

In Dataflowm it parallels tasks automatically

The code is the same between real-time(stream) and batch

### Apache Beam pipelines
Source -> Transforms (series of steps) -> Sink (out of the pipeline)
* In each transform, the input and output will be a data structure **PCollection**
* The pipeline is executed on the cloud by a Runner
    * each Runner is unique to each platform
```
    import apache_beam as beam

    if __name__ == '__main__':
        # create a pipeline parameterized by command line flags
        p = beam.Pipeline(argv = sys.argv)

        (p
            | 'Read' >> beam.io.ReadFromText('gs://...')    # read input
            | 'CountWords' >> beam.FlatMap(lambda line: count_words(line))
            | 'Write' >> beam.io.WriteToText('gs://...')    # write output
        )

    p.run()
```
* Once the pipeline instance is created, every transform is implemented as an argument to the applied method of the pipeline.
* '|' operator is overloaded to call the apply method
* **PCollection** is a data structure with pointers that points to where the data flow cluster stores your data.

LAB: training-data-analyst/courses/data_analysis/lab2/python/grep.py


## Scale Data Pipeline

### MapReduce
MapReduce approach splits Big Data so that each compute node processes data local to it
* The data will be sharded into multiple cluster nodes
* Map is a stateless function, so that it can be scheduled to each cluster node and apply on the data in each node
* The result of each Map will be shuffled and do Reduce and get the result

### Pardo
If you want to take a transformation in your data processing pupeline and let Dataflow run it at scale with automatic distribution across many odes in a cluster, you should use the **Apache Beam Pardo Class**
* It allows parallel processing
* It acts one item at a time (like **Map** in MapReduce)
    * multiple instances of class on many machines
    * should not contain any state
* Useful for:
    * Filtering (choosing which inputs to emit)
    * Extracting parts of an input (e.g. fields of TableRow)
    * Converting one Java type to another
    * Calculating values from different parts of inputs

### Map vs FlatMap
* Use Map for 1:1 relationship between input and output
```
    'WordLengths' >> beam.Map(lambda word: (word, len(word)))
```

* FlatMap for non 1:1 relationships, usually with generator
```
    def vowels(word):
        for ch in word:
            if ch in ['a', 'e', 'i', 'o', 'u']:
                yield ch

    'WordVowels' >> beam.Flatmap(lambda word: vowels(word))
```

### GroupBy
It is an operation that do shuffling
In Dataflow, shuffle explicitly with a GroupByKey
Create a Key-Value pair in ParDo
Then Group by the key

* Similar to **shuffle** in MapReduce
```
    cityAndZipCodes = p
        | beam.Map(lambda address: (address[1], address[3]))
        | beam.GroupByKey()
```

### Combine.PerKey
It lets you aggregate

* Can be applied to a PCollection value
```
    totalAmount = salesAmounts | Combine.globally(sum)
```
* And also th a grouped Key-Valued pair:
```
    totalSalesPerPerson = salesRecords | Combine.perKey(sum)
```

LAB: training-data-analyst/courses/data_analysis/lab2/python/is_popular.py


## Preprocessing with Cloud Dataprep
There are 2 general approaches to design preprocessing

### Cloud Datalab
1. Explore in Cloud Datalab
2. Write code in BigQuery / Dataflow / Tensorflow to transform data

You can use SQL and calculate statistics using BigQuery
or use Beam and Dataflow to do the same thing

### Cloud Dataprep
1. Explore in Cloud Dataprep
2. Design Recipe in UI to preprocess data
3. Apply generated Dataflow transformations to all data
4. Reuse Dataflow transformation in real-time pipeline

It supports the full preprocessing lifecycle
Wranglers write beam code in Dataflow (automatically)

LAB: Learn how to use Dataprep


## Feature Crosses
* The feature cross provides a way to combine features to make it fit in a linear model

* It is important to choose the decision boundary so that you don't need too much additional parameters (Discretize)

* Separate prediction per grid cell
* The weight of a cell is essentially the prediction for that cell

* A feature cross **memorizes** the input space
* Goal of ML is **generalization**
* Memorization works when you have lots of data

* Feature Cross brings a lot of power to linear models
* Feature Cross + Massive Data is an efficient way for learning highly complex spaces
* Feature Cross allows a linear model to memorize large datasets
* Optimizing linear models is a convex problem

LAB: [A Neural Network Playground](http://playground.tensorflow.org/#activation=relu&batchSize=5&dataset=xor&regDataset=reg-plane&learningRate=0.01&regularizationRate=0&noise=35&networkShape=&seed=0.22297&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
LAB: [A Neural Network Playground](http://playground.tensorflow.org/#activation=relu&batchSize=5&dataset=circle&regDataset=reg-plane&learningRate=0.01&regularizationRate=0&noise=35&networkShape=&seed=0.92217&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

* Feature Cross combines discrete / categorical features
* It leads the input to be very sparse

* Crossing is possible with categorical or discretized columns

LAB: [A Neural Network Playground](http://playground.tensorflow.org/#activation=relu&batchSize=1&dataset=gauss&regDataset=reg-plane&learningRate=0.01&regularizationRate=0&noise=30&networkShape=&seed=0.48621&showTestData=false&discretize=false&percTrainData=30&x=true&y=true&xTimesY=true&xSquared=true&ySquared=true&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

* One reason of overfitting is because we are giving the same data in multiple ways

* If we use a model with too many feature crosses, which makes the model too complicated for only simple data -> it may fit to the noise in the training data
-> use *regularization* to remove features


## Implement Feature Cross
