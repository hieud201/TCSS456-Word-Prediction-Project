# import statements
import os
import string
import re
import sys
from pickle import load
from random import randint

import numpy as np
from keras import Sequential
from keras.src.layers import Embedding, LSTM, Dense
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from tensorflow.python.keras.models import load_model

TEXTS_PATH = '../data/'


def main():
    text_docs: list[str] = load_docs()
    cleaned_tokens: list[str] = clean_docs(text_docs)
    sequences = organize_into_sequences(cleaned_tokens)





    with open("tokens.txt", "w") as file:
        file.write('\n'.join(sequences))

    print(cleaned_tokens)


def load_docs() -> list[str]:
    docs: list[str] = []
    for file in os.listdir(TEXTS_PATH):
        with open(TEXTS_PATH + file, "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs


def clean_docs(docs: list[str]) -> list[str]:
    def clean_doc(doc: str):
        # replace  --  with a space
        doc = doc.replace('--', ' ')
        # split into tokens by white space
        tokens = doc.split()
        # prepare regex for char filtering
        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        # remove punctuation from each word
        tokens = [re_punc.sub('', w) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # make lower case
        tokens = [word.lower() for word in tokens]
        return tokens

    out = []
    for doc in docs:
        out += clean_doc(doc)
    return out


def organize_into_sequences(tokens: list[str]) -> list[str]:
    length = 50 + 1
    sequences = []
    for i in range(length, len(tokens)):
        seq = tokens[i - length:i]
        line = ' '.join(seq)
        sequences.append(line)
    return sequences


def save_doc(lines: list[str], filename):
    data = '\n'.join(lines)
    with open(filename, 'w') as file:
        file.write(data)


def create_tokenizer(lines: list[str]):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    # integer encode sequences of words
    sequences = tokenizer.texts_to_sequences(lines)
    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1

    # separate into input and output
    sequences = array(sequences)
    X, y = sequences[:, :-1], sequences[:, -1]
    y = to_categorical(y, num_classes=vocab_size)
    seq_length = X.shape[1]



main()










# TO DO: define the model based on architecture provided in the document and follow the process as seen in Word-based LM model implemented in class
def define_model(vocab_size, seq_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    # Note: for the first LSTM (i.e. lstm_1 layer set additional parameter: return_sequences=True
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    # Note: for dense_1, activation = 'relu' and for dense_2,  activation = 'softmax'
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


# define model
# model = define_model(vocab_size, seq_length)
# fit model
# model.fit(X, y, batch_size=128, epochs=100)

# Note: this will take some time, hence we will save the model and load later
# save the model to file.
# Note: Once the model has been saved you can comment out the model section (so you don't have to run 100 epochs)
# model.save(DRIVE_DIR + 'model.h5')
# save the tokenizer
# dump(tokenizer, open(DRIVE_DIR + 'tokenizer.pkl', 'wb'))


# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = np.argmax(model.predict(encoded, verbose=0), axis=-1)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)


# load cleaned text sequences
doc = load_doc(tokens_file)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

# load the model
model = load_model(TEXTS_PATH + 'model.h5')
# load the tokenizer
tokenizer = load(open(TEXTS_PATH + 'tokenizer.pkl', 'rb'))
# select a seed text
seed_text = lines[randint(0, len(lines))]
print(seed_text + '\n')
# generate new text
generated = generate_seq(model, tokenizer, seq_length, seed_text, 500)
print("Generated:")
print(generated)
