# Lesson 3: Natural Language Processing

## Encode Sentence
* from **keras.preprocessing.text** import **Tokenizer**
* Generate a dictionary of word encodings
* Create vectors out of sentences
```
    sentences = ['asdfas ld;if j;l', 'asdf']
    
    # num_words is the number of unique words in your dictionary
    # Tokenizer will take #num_words of most popular words in the sentence
    tokenizer = Tokenizer(num_words=)
    tokenizer.fit_on_texts(sentences)   # build corpus from sentences
```
* Get word indices
* Turn sentences into indices
```
    word_index = tokenizer.word_index                   # Get the index Tokenizer gives
    sequence = tokenizer.texts_to_sequences(sentences)  # it will only show the indices which are in the dictionary

```

### Index the words not in your corpus dictionary
* Add a property **oov_token** when instantiate Tokenizer
* All unseen words will be indexed as '<OOV>'
```
    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
```


## Padding
* We want our inputs in uniform size
* from **keras.preprocessing.sequence** import **pad_sequences**
* pad 0 to make all sentences same length
```
    padded = pad_sequences(sequences, padding='post', maxlen=5, truncating='post')
```
* Arguments in pad_sequences:
    * padding: pad 0 in front(*default*)/after the sentence to **maxlen**
    * maxlen: limit the sentence length (default = longest sentence)
    * truncating: if a sentence is longer than **maxlen**, it will truncate from beginning(*default*)/end


## Use Tensorflow Dataset
* if using **Tensorflow 1.x**: add **tf.enable_eager_execution()**
* pip install -q tensorflow-datasets
* import **tensorflow-datasets** as tfds
* turn the data into **numpy** arrays
    * s.numpy()
    * np.array(training_sentences)


## Text Sentiment Embedding
* Construction of word vector is based on the label of your data, so that the corpus will cluster together with more positive/negative
* Use **tf.keras.Embedding()**
    * vocab_size
    * embedding_dim
    * input_length
    * The result of Embedding layer will be a 2D array
        *  (len(sentence), embedding_dim)
* In NLP, the **Flatten Layer** may make the ouput size too large
    * Use **GlobalAveragePooling1D()** instead


## Use SubwordTextEncoder
* It is tokenized in subwords, and it is case-sensitive
```
    imdb, info = tfds.load('.../subwords8k', ...)
    tokenizer = info.features['text'].encoder
    tokenizer.subwords                              # see subwords
    tokenizer.encode(sentence)                      # encode sentence
    tokenizer.decode(tokenized_string)              # decode tokenized_string
```


## LSTM
* Add a layer **tf.keras.Bidirectional(tf.keras.layers.LSTM(64))**
    * Cell state means that they carry context along with them
    * 64 = cell_state is the number of **outputs** that I desired from this layer
    * **Bidirectional** can make the cell states go in both direction
        * Words will have two directional meanings
            * ex: *big dog* and *dog big* are both meaningful
        * It will **double** the outputs from LSTM
* If you want to stack one LSTM layer on another LSTM layer
    * Add parameter **return_sequences = True**
    * Ensure that the outputs of the LSTM match the desired inputs of the next LSTM layer

```
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), return_sequences=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(),
        tf.keras.layers.Dense()
    ])
```


## Another RNN - Gated Recurrent Unit (GRU)
* Use **tf.keras.Bidirectional(tf.keras.layers.GRU())**

## Use Convolution on Texts
* Use **Conv1D(No.Convs, size, activation=)**
    * size = how many words in a convolution
```
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(64),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(),
        tf.keras.layers.Dense()
    ])
```

## Generate Texts
Use different n-grams to make inputs. 
Take the sentence to be **"input"** and the last word to be **"label"**, then do classification
* Tokenize the text line by line
* For each line, generate a list of n-grams (start from 2)
* Ex: 'you are a pig'
```
    sentence = 'you are a pig'
    token = tokenizer.encode(sentence) == [2, 5, 6, 8]
    ngrams = [
        [2, 5],
        [2, 5, 6],
        [2, 5, 6, 8]
    ]

    # pad ngrams (in pre position)
    ngrams = [
        [0, 0, 2, 5],
        [0, 2, 5, 6],
        [2, 5, 6, 8]
    ]
```

* Take the **last element** of each row as **labelY**, the rest is labelX
```
    labelX = [0, 0, 2]      labelY = 5
             [0, 2, 5]               6
             [2, 5, 6]               8
```

* Use **tf.keras.utils.to_categorical(labels, num_classes=total_words)**
    * To turn the label into a list with *total_words* length
    * Ex: total_words = 8
```
    total_words = 8
    ys = tf.keras.utils.to_categorical(5, num_classes=total_words)
    # labelY = 5
    # ys = [0, 0, 0, 0, 1, 0, 0, 0]
```

### Character-based RNN
[Text generation using a RNN with eager execution  |  TensorFlow Core](https://www.tensorflow.org/tutorials/sequences/text_generation)

## Regularization
* from **tf.keras** import **regularizers**
* regularizers.l1, regularizers.l2, regularizers.l1_l2
* **L1** is the **sum** of the weights
* **L2** is the **square sum** of the weights
* It adds a **regularization** term on a **layer** in order to prevent the coefficients to overfit