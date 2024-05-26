data = """Jack and Jill went up the hill\n
To fetch a pail of water\n
Jack fell down and broke his crown\n
And Jill came tumbling after\n"""

print(data)


import tensorflow as tf

# two-word input to one-word output
from tensorflow.keras.preprocessing.text import Tokenizer
from numpy import array
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import numpy as np

# define our model
def define_model(X):
  model = Sequential()
  model.add(Embedding(vocab_size, 10, input_length=max_length - 1))
  model.add(LSTM(50))
  model.add(Dense(vocab_size, activation='softmax'))
  # compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  # summarize defined model
  model.summary()
  return model


# Generate a sequence from the model
def generate(model, tokenizer, max_length, seed_text, n_words):
  in_text = seed_text   # Input word
  result = seed_text    # Start of the output

  # Generate a fixed number of words
  for _ in range(n_words):
    # Encode text as integer
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    # Convert the encoded test integer to array
    encoded = array(encoded)
    # Prepad the sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen = max_length, padding="pre")
    # Predict the next word
    yhat = np.argmax(model.predict(encoded, verbose = 0), axis = -1)
    # Map the predicted index to word
    out_word = ''

    for word, index in tokenizer.word_index.items():
      if yhat == index:
        out_word = word
        break

    # Append to the input
    in_text, result = out_word, result + ' ' + out_word

  return result

# TRAINING PROCESS
# Encoding
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
# Convert the sequences of text (words) into sequences of integers
encoded = tokenizer.texts_to_sequences([data])[0]

print(encoded)

# Determine vocab size
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)


# Create 2 words -> word sequences
sequences = list()
for i in range(2, len(encoded)):
  sequence = encoded[i - 2 : i + 1]
  sequences.append(sequence)

print("Total sequences:", len(sequences))
print(sequences)


# Pad the sequence
max_length = max([len(seq) for seq in sequences])
print(max_length)

sequences = pad_sequences(sequences, maxlen = max_length, padding="pre")

# Split Inpus & Ouput
sequences = array(sequences)
print(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]
print('Input:', X, 'Output:', y)


# Convert output to one-hot vector representation
y = to_categorical(y, num_classes=vocab_size)

# Define the model
model = define_model(vocab_size)
# Fit the model
model.fit(X, y, epochs=300, verbose=2)


# Testing process
# evaluate model
print(generate(model, tokenizer, max_length - 1, "Jack and", 5))
print(generate(model, tokenizer, max_length - 1, "fell down", 3))
print(generate(model, tokenizer, max_length - 1, "broke and", 5))
