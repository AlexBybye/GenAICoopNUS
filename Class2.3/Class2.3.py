import tensorflow as tf

from tensorflow import keras

# load in the libraries and define the stopwords

import tensorflow_datasets as tfds

import numpy as np

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
             "yourselves"]
# load in the training and test data
# we were using the training split earlier; the dataset also has a test split
train_set = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))
test_set = tfds.as_numpy(tfds.load('imdb_reviews', split = "test"))
# we use a TextVectorization layer for cleaning and tokenization

# further import TextVectorization class, re and string modules
from tensorflow.keras.layers import TextVectorization
import re
import string


def custom_standardization(input_data):
    # make all lowercase
    data = tf.strings.lower(input_data)
    # remove punctuation
    data = tf.strings.regex_replace(
        data, f"[{re.escape(string.punctuation)}]", ""
    )

    # remove HTML tags
    data = tf.strings.regex_replace(data, '<br />', ' ')

    # remove stopwords
    for i in stopwords:
        data = tf.strings.regex_replace(data, f" {i} ", " ")
    return data


# we also define our customized split function
# if no "sep" is provided as the argument for the split function, we split according to whitespace by default
def custom_split(input_data):
    return tf.strings.split(input_data)

# instantiate our text vectorization layer
# just set output_mode to "int" to assign a unique integer to each word
text_vectorization = TextVectorization(
    output_mode = 'int',
    standardize = custom_standardization,
    split = custom_split,
    # we set the vocab size to 20000 and max length of review to 600 words
    max_tokens = 20000,
    output_sequence_length = 600
    )
# we extract out the text reviews and labels from the train and test sets
# besides the text portion containing the reviews, there is also a label portion classifying each review
# as positive (1) or negative (0)

# create an empty list to store all the reviews and their labels later
train_imdb_reviews = []
train_labels = []
test_imdb_reviews = []
test_labels = []

for review in train_set:
    train_imdb_reviews.append(str(review['text']))
    train_labels.append(int(review['label']))

for review in test_set:
    test_imdb_reviews.append(str(review['text']))
    test_labels.append(int(review['label']))
# check the first test review
print(test_imdb_reviews[0])

# Read through the review, is it positive or negative?
# let's check the label
print(test_labels[0])

# Do we see the expected result?
# we have our train and test sets; how about the validation set?
# out of the train set, we select 20% out for validation
validation_size = int(0.2 * len(train_imdb_reviews))
print(validation_size)
# create the actual train and validation set

# the 0th review to the 5000th review (exclusive) forms our validation set; the rest goes into training
actual_train_imdb_reviews = train_imdb_reviews[validation_size:]
val_imdb_reviews = train_imdb_reviews[0:validation_size]

actual_train_labels = train_labels[validation_size:]
val_labels = train_labels[0:validation_size]

# size of our train set
print(len(actual_train_imdb_reviews))
# size of our validation set
print(len(val_labels))
# learn the vocabulary by calling the adapt() method
# remember that we should only be "seeing" the actual train dataset at this stage
text_vectorization.adapt(actual_train_imdb_reviews)

# subsequently we convert each review to a sequence of indexed tokens, for train, val and test sets
vect_train_reviews = text_vectorization(actual_train_imdb_reviews)
vect_val_reviews = text_vectorization(val_imdb_reviews)
vect_test_reviews = text_vectorization(test_imdb_reviews)

# finally, we see the first review, padded up to a max length of 600
print(vect_train_reviews[0])
# look at the learned vocabulary
text_vectorization.get_vocabulary()
# check the type of data storage for our reviews
print(type(vect_train_reviews))
# Need to convert to numpy arrays first, which is used by TensorFlow for training

arr_train_reviews = np.array(vect_train_reviews)
arr_train_labels = np.array(actual_train_labels)

arr_val_reviews = np.array(vect_val_reviews)
arr_val_labels = np.array(val_labels)

arr_test_reviews = np.array(vect_test_reviews)
arr_test_labels = np.array(test_labels)
# check the new data storage structure
print(type(arr_train_reviews))

print(arr_train_reviews.shape,arr_train_labels.shape,arr_val_reviews.shape,arr_val_labels.shape,arr_test_reviews.shape,arr_test_labels.shape)
# we have the corresponding 20000 labels for each review in our train set
# we will build a function for our bag-of-words model so that we can easily reuse it later

# importing the layers API from keras
from tensorflow.keras import layers

# setting the max no of tokens in a review; previously we only specified within our text vectorization object
max_length = 600
# we define the number of neurons in our layer
layer_dim = 16


# define our function, which is going to take in two arguments
# the input shape, defined as vectors of 600 elements each, as well as the dimension of each layer
def get_model(input_shape=max_length, num_dim=layer_dim):
    # recall the model that we built in our first class; let's go to our slides and go through this line by line
    inputs = keras.Input(shape=(input_shape,))
    x = layers.Dense(num_dim, activation='relu')(inputs)

    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
# let's take a look at the model we have configured, by running summary()
model = get_model()
model.summary()

# we see indeed that the number of trainable parameters is 9616 and 17 respectively for our intermediate
# and output layers
# the number of epochs to train for
num_epochs = 100

# let's go back to our slides to discuss the rest of the code here
callbacks = [
    keras.callbacks.ModelCheckpoint('bag_of_words_model', save_best_only = True)
]

model.fit(arr_train_reviews, arr_train_labels, epochs = num_epochs, batch_size = 32,
         validation_data = (arr_val_reviews, arr_val_labels), callbacks = callbacks)

# we notice that for each epoch, we have the number 625, indicating that we are updating the weights
# 625 times per epoch
# We have the time taken to complete each epoch
# The loss and accuracy at each training epoch and for the validation set are also reported
# We notice that the losses are relatively high at around 0.75 for the validation set, and an
# accuracy of 0.49
# load our saved model from training
model = keras.models.load_model('bag_of_words_model')

# we apply the model on the test set, using evaluate()
print(f"Test acc: {model.evaluate(arr_test_reviews, arr_test_labels)[1]:.3f}")

# How do we feel about the test accuracy reported? <Go to slides>