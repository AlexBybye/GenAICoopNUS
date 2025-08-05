# import tensorflow and keras
import tensorflow as tf
from tensorflow import keras

# import the Tokenizer class from the keras preprocessing library for word tokenization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# we can then simply call the function to convert our sequences into a padded set

# we have two sample sentences, stored in the variable "sentences"
sentences = [
'I am very happy today',
'I am very sad today',
'Am I happy today?',
'I feel happy taking a walk outside today'
]

# create a Tokenizer object (instantiating a class gives an object)
# and specify the number of words it can tokenize
# note that setting num_words = 100 means that the most common 99 words will be kept, i.e. num_words - 1
# we have a very small corpus in our sample sentences, so no issue with the limit here

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")

# generate our word index
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)

print(sequences)

# we apply our word index again on our test sentences, to vectorize them
test_sequences = tokenizer.texts_to_sequences(sentences)
print(word_index)
print(test_sequences)
# save our word index, by accessing the word_index attribute of our tokenizer object
word_index = tokenizer.word_index
# let's take a look at our word index

# padded = pad_sequences(sequences, padding = 'post')
# print(padded)

padded = pad_sequences(sequences, padding='post', maxlen=5)
print(padded)

with open("RESULTS/Case5.txt", "w", encoding="utf-8") as f:
    for word, index in word_index.items():
        f.write(f"{word}: {index}\n")

