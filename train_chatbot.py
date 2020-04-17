import nltk
from nltk import WordNetLemmatizer

import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)  # Tokenize each words
        words.extend(w)
        documents.append((w, intent['tag']))  # Add documents in the corpus

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]  # Lemmatize, lower each word
words = sorted(list(set(words)))  # Remove duplicates in words
classes = sorted(list(set(classes)))  # Remove duplicates in classes

print(len(documents), "documents")
print(len(classes), "classes")
print(len(words), "unique lemmatize words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Training the data
training = []
# Create an empty array for our output
output_empty = [0] * len(classes)

# Training set, bag of words for each sentence
for doc in documents:
    # Initialize our bag of words
    bag = []
    # List of tokenized words for the pattern
    pattern_word = doc[0]
    # Lemmatize each word - create base word, in attempt to represent related words
    pattern_word = [lemmatizer.lemmatize(word.lower()) for word in pattern_word]

    # Create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_word else bag.append(0)

    # Output is a '0' for each tag and '1' for current tag ( for each pattern )
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Random shuffle our features and turn into np.array using numpy
random.shuffle(training)
training = np.array(training)

# Create train and test lists, X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# Making neural network
model = Sequential()
# First layer using 128 neurons
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
# Second layer using 64 neurons
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# Third layer contains number of neurons
# Equal the number of intents to predict output intents with sofmax
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic Gradient Descent with Nesterov accelerated gradient gives good result for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("Model created")
