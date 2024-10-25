import nltk

# Download a package from nltk
# nltk.download('punkt')   # package w/ a pre-trained tokenizer

# Stemming: reduce a word to its root form
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

import numpy as np


# Tokenize: split a sentence into a list of words
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag =   [  0,      1,      0,    1,      0,      0,       0]
    """

    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
            
    return bag