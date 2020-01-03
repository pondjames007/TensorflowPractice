import subprocess

subprocess.check_output(['wget', '--no-check-certificate', 
'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv', 
'-O', './bbc-text.csv'])


import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = 0.8

sentences = []
labels = []
stopwords = [ 
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", 
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", 
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", 
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", 
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", 
    "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", 
    "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", 
    "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", 
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", 
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", 
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until", 
    "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", 
    "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", 
    "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

with open("/tmp/bbc-text.csv", 'r') as csvfile:
    # YOUR CODE HERE
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
        sentences.append(sentence)

# Separate sentences into Train and Test dataset
train_size = int(len(sentences) * training_portion)
train_sentences = sentences[:train_size]
train_labels = labels[:train_size]

validation_sentences = sentences[train_size:]
validation_labels = labels[train_size:]

print(train_size)               # 1780
print(len(train_sentences))     # 1780
print(len(train_labels))        # 1780
print(len(validation_sentences))# 445
print(len(validation_labels))   # 445

# Tokenize the sentences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

# Change the sentences into indices and pad them into same length
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, paddin=padding_type, maxlen=max_length)

print(len(train_sequences[0]))  # 449
print(len(train_padded[0]))     # 120

print(len(train_sequences[1]))  # 200
print(len(train_padded[1]))     # 120

print(len(train_sequences[10])) # 192
print(len(train_padded[10]))    # 120

validation_sequences = tokenizer.texts_to_sequences(validation_sentences) # YOUR CODE HERE
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length) # YOUR CODE HERE

print(len(validation_sequences))# 445
print(validation_padded.shape)  # (445, 120)


label_tokenizer = Tokenizer() # YOUR CODE HERE
label_tokenizer.fit_on_texts(labels) # YOUR CODE HERE

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels)) # YOUR CODE HERE
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels)) # YOUR CODE HERE

print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)

print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 30
history = model.fit(
    train_padded, 
    training_label_seq, 
    epochs=num_epochs, 
    validation_data=(validation_padded, validation_label_seq), 
    verbose=2

)