"""
Week1: Encode Sentence
"""

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

# add this if tensorflow version < 2.0
tf.enable_eager_execution()


imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']
training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []

for s, l in train_data:
    print(type(s), type(l))
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())