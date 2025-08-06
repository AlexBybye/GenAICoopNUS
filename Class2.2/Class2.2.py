import tensorflow as tf
from tensorflow import keras
# import tfds that contains our IMDB movie reviews dataset
import tensorflow_datasets as tfds
import numpy as np

# load in the dataset, specifically the "train" split
# using tfds.as_numpy ensures that we load the data as iterable NumPy arrays
train_set = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))

# check how many reviews we have
len(train_set)

# create an empty list to store all the raw reviews
raw_imdb_reviews = []

# we extract the 'text' portion, which contains the review
for review in train_set:
    # append the review to my empty list
    raw_imdb_reviews.append(str(review['text']))

# we take a look at the first review, prior to cleaning
# Python indices start from 0

raw_imdb_reviews[1]

# import the packages for removing punctuation and html tags
# the BeautifulSoup class

from bs4 import BeautifulSoup
# the string module
import string

# we define a list of of stopwords that we wish to remove
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
# this cell does the data cleaning
# will take some time to run (~10-15 mins - be patient!)

# create an empty list to store all the cleaned reviews later
imdb_reviews = []

# create an empty string variable to aggregate all our movie reviews
big_imdb_reviews = ''

# define a translation table for use later, in this case to remove any punctuation
# as specified in the third argument
table = str.maketrans('', '', string.punctuation)

# go through each review in our list
for review in train_set:
    # convert all to lowercase using the lower() method
    line = str(review['text'].decode('UTF-8').lower())

    # convert to BeautifulSoup object
    soup = BeautifulSoup(line)
    # extract the text component only and remove the HTML tags, using the get_text() method
    line = soup.get_text()

    # split each review into individual words
    words = line.split()

    # For each word, remove punctuation
    # and if it isn't a stopword, add to the string, filtered_review
    filtered_review = ""
    for word in words:
        # using the translate() method, pass in our mapping table defined earlier to remove punctuation
        word = word.translate(table)
        if word not in stopwords:
            filtered_review = filtered_review + word + " "

    # at this point, we are going to have our cleaned review, without punctuation and stopwords
    # add the cleaned review to our list
    imdb_reviews.append(filtered_review)

    # add the cleaned review to our big aggregated review
    big_imdb_reviews += filtered_review

imdb_reviews[0]
# comparing the result here with the result from before, we see that it's much better here
# with respect to standardisation, removal of punctuation, HTML tags, stopwords
# since we are preparing the text for predicting sentiment, that it is less readable is not as
# much of a concern (more on that later)
# we look at the first 1000 characters in our big aggregated movie review
big_imdb_reviews[:1000]
# we set a limit of 20000 words for our vocabulary, and specify the OOV token
# Recall that this means we keep the 19999 most common words, including the OOV token
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000, oov_token = '<OOV>')

# we tokenize and index each word in our dataset
tokenizer.fit_on_texts(imdb_reviews)

# Our tokenizer.word_index is a dictionary. We call items() method to extract the items in this dictionary
# for subsequent conversion to a list, and display the first 10 items in the list using print()
# this is for us to see the first few items in our word index, rather than all 20000
print(list(tokenizer.word_index.items())[:10])

# we see the word and its associated index
# convert each review to a sequence of indexed tokens, i.e. vectorization
sequences = tokenizer.texts_to_sequences(imdb_reviews)

# print the tokenized first review
print(sequences[0])
# let's check how many words there are in our indexed vocabulary
len(tokenizer.word_index)

# Qn 5: Anything unusual we notice here? <see slides>

# put our aggregated review into a list for vectorization
list_big_imdb_reviews = [big_imdb_reviews]

# vectorize our aggregated review
aggregated_sequence = tokenizer.texts_to_sequences(list_big_imdb_reviews)

# check the number of unique tokens in our aggregated review
len(np.unique(aggregated_sequence))

# we see indeed that only 19999 unique tokens have been used to vectorize our sequences
# Qn 6: Why isn't the limit imposed when we tokenize and index our corpus,
# but only when we are vectorizing? <slides>
# to have the option of changing the word limit without having to fit to the corpus again
# we need to specify the limit separately

# create our tokenizer object
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token = '<OOV>')

# we fit to our corpus
tokenizer.fit_on_texts(imdb_reviews)

# put our aggregated review into a list for vectorization
list_big_imdb_reviews = [big_imdb_reviews]

# here, before we vectorize, we set the limit on the number of tokens we want to use
tokenizer.num_words = 10000

# vectorize our aggregated review
aggregated_sequence = tokenizer.texts_to_sequences(list_big_imdb_reviews)

# check the number of unique tokens in our aggregated review
len(np.unique(aggregated_sequence))

# Do we see the expected result below?
# say we are unhappy with the limit, and wish to up it to 20000
# we can just do so by changing the num_words attribute, without re-fitting the corpus
tokenizer.num_words = 20000

aggregated_sequence = tokenizer.texts_to_sequences(list_big_imdb_reviews)
len(np.unique(aggregated_sequence))
# length of the longest review
len(max(sequences, key = len))
# find the location of the longest review, i.e. is it 5th review, the 90th review, etc
# create an empty list to store the result
long = []

# going through each review
for i in range(len(sequences)):
    # if the length of review matches the length of the longest review, then store its index into long[]
    if len(sequences[i]) == 1430:
        long.append(i)

long
# double check the review with index 3108
len(sequences[3108])
# import the tool needed for padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

padding = pad_sequences(sequences)
print(padding)

# we see the padding is done from the start as default
padding = pad_sequences(sequences, padding = 'post')
print(padding)

# now we see that the padding is done at the end
# now all the sequences are the same shape, with 1430 tokens
padding.shape
# sometimes we might just want to set a max number of words for each review
# since the sentiment or gist of each review can probably be gleaned from the first few words
padding = pad_sequences(sequences, padding = 'post', maxlen = 600)
print(padding)
# we see that each of the 25000 rows has 600 tokens
padding.shape
# the truncating is actually done from the start, for reviews longer than 600 words
# this is the index for the last word of the longest review (untruncated)
sequences[3108][1429]
# checking the truncated review, we see that the last word of the review 10194 still included
# showing the truncating is at the start
padding[3108]
# now often the most important part of the review is at the start
# so if there's any truncating, it should probably be done from the end of the review
padding = pad_sequences(sequences, padding = 'post', maxlen = 600, truncating = 'post')
print(padding)
# checking the truncated longest review again
padding[3108]
# we see that the first word of the review is included in the truncated review at the top
# showing that the truncating is now at the back.
sequences[3108][0]

# Let's go to our slides first for a brief discussion on the layer

# import the TextVectorization class

from tensorflow.keras.layers import TextVectorization

# we can also define custom functions for standardization as well
# for instance, let's define the following customization function

# import the necessary modules
import re
import string


def custom_standardization(input_data):
    # make all lowercase
    data = tf.strings.lower(input_data)
    # remove punctuation using regex_replace() function
    data = tf.strings.regex_replace(
        data, f"[{re.escape(string.punctuation)}]", ""
    )

    # remove HTML tags; see the code chunk below for an example of the tags we want to remove
    data = tf.strings.regex_replace(data, '<br />', ' ')

    # remove stopwords
    for i in stopwords:
        # we need the space before and after, so we don't accidentally replace say "i" in the middle of a word
        data = tf.strings.regex_replace(data, f" {i} ", " ")
    return data


# we can also define a customized split function
# if no "sep" is provided as the argument for the split function, we split according to whitespace by default
def custom_split(input_data):
    return tf.strings.split(input_data, sep=" ")
raw_imdb_reviews[2]
# note the <br /> tags
# so now we can clean and tokenize all in one step!
# create our TextVectorization object
# setting the output_mode to "int" to assign a unique integer to each word
text_vectorization = TextVectorization(
    output_mode = 'int',
    standardize = custom_standardization,
    split = custom_split,
    # we set the vocab size to 20000 and max length of review to 600 words
    max_tokens = 20000,
    output_sequence_length = 600
    )
# verify that the type of data storage structure for our raw reviews, is a list
type(raw_imdb_reviews)
# we call the adapt() method to learn the vocabulary
text_vectorization.adapt(raw_imdb_reviews)

# subsequently we convert each review to a sequence of tokens, by passing them into the text
# vectorization object / layer
vectorized_reviews = text_vectorization(raw_imdb_reviews)

# finally, we see the first review
vectorized_reviews[0]

# remember that the cleaning steps are already defined in my text vectorization object
# we see that it has 600 elements for each row
vectorized_reviews.shape
# let's look at the vocabulary learned
text_vectorization.vocabulary_size()

# expected result, since we set max tokens to be 20000
# we retrieve our learned vocabulary from adapt() using the get_vocabulary() method
vocabulary = text_vectorization.get_vocabulary()

vocabulary