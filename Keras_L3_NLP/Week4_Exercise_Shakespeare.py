from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
import numpy as np

tokenizer = Tokenizer()

import subprocess

subprocess.check_output(['wget', '--no-check-certificate', 
'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt', 
'-O', './sonnets.txt'])

data = open('./sonnets.txt').read()
corpus = data.lower().split('\n')

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1


# Create input sequences using list of tokens
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# pad sequences
max_sequences_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequences_len, padding='pre'))

# create predictors and label
predictors, label = input_sequences[:,:-1], input_sequences[:-1]
label = ku.to_categorical(label, num_classes=total_words)


# Build the model
model = Sequential()
model.add(Embedding(total_wrods, 100, input_length=max_sequences_len-1))
model.add(Bidirectional(LSTM(150, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(total_words, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(predictors, label, epochs=100, verbose=1)
