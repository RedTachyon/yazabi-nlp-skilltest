# Copyright (c) 2017 Yazabi Predictive Inc.

#################################### MIT License ####################################
#                                                                                   #
# Permission is hereby granted, free of charge, to any person obtaining a copy      #
# of this software and associated documentation files (the "Software"), to deal     #
# in the Software without restriction, including without limitation the rights      #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell         #
# copies of the Software, and to permit persons to whom the Software is             #
# furnished to do so, subject to the following conditions:                          #
#                                                                                   #
# The above copyright notice and this permission notice shall be included in all    #
# copies or substantial portions of the Software.                                   #
#                                                                                   #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR        #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,          #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE       #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER            #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,     #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE     #
# SOFTWARE.                                                                         #
#                                                                                   #
#####################################################################################

# This module contains the function signatures for data preprocessing on the IMDB
# movie review dataset. The code is heavily commented to allow you to follow along easily.

# Please report any bugs you find using @yazabi, or email us at contact@yazabi.com.
from __future__ import print_function
from __future__ import generators
from __future__ import division
import os, sys
from tqdm import tqdm
import numpy as np
# useful packages
import nltk
import gensim
import sklearn


def load_imdb_data():
    """
    Load the IMDB data.

    Returns:
        train_data, test_data: lists of reviews (strings)
        train_labels, test_labels: arrays of binary labels
    """
    pos_path = [os.path.join('aclImdb', 'train', 'pos', pth) for pth in os.listdir('aclImdb/train/pos')]
    neg_path = [os.path.join('aclImdb', 'train', 'neg', pth) for pth in os.listdir('aclImdb/train/neg')]

    test_pos_path = [os.path.join('aclImdb', 'test', 'pos', pth) for pth in os.listdir('aclImdb/test/pos')]
    test_neg_path = [os.path.join('aclImdb', 'test', 'neg', pth) for pth in os.listdir('aclImdb/test/neg')]

    pos_data = []
    for path in tqdm(pos_path):
        with open(path, 'r') as f:
            data = f.read()
            pos_data.append(data)

    neg_data = []
    for path in tqdm(neg_path):
        with open(path, 'r') as f:
            data = f.read()
            neg_data.append(data)

    test_pos_data = []
    for path in tqdm(test_pos_path):
        with open(path, 'r') as f:
            data = f.read()
            test_pos_data.append(data)

    test_neg_data = []
    for path in tqdm(test_neg_path):
        with open(path, 'r') as f:
            data = f.read()
            test_neg_data.append(data)

    train_labels = np.hstack((np.ones([len(pos_data)]), np.zeros([len(neg_data)])))
    test_labels = np.hstack((np.ones([len(test_pos_data)]), np.zeros([len(test_neg_data)])))

    return pos_data + neg_data, test_pos_data + test_neg_data, train_labels, test_labels


def tokenize(text):
    """Tokenize and filter a text sample.
    Hint: nltk

    Args:
        text: string to be tokenized and filtered.

    Returns:
        tokens: a list of the tokens/words in text.
    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(text.lower())


def clean_data(data):
    """
    Clean the data to remove punctuation etc.
    """
    token_data = list(map(tokenize, data))
    return list(map(lambda x: ' '.join(x), token_data))


def make_embedding_matrix(texts, size):
    """Create an embedding matrix from a list of text samples.
    Hint: gensim

    params:
    :texts: a list of text samples containing the vocabulary words.
    :size: the size of the word-vectors.

    returns:
    :embedding_matrix: a dictionary mapping words to word-vectors (embeddings).
    """
    token_texts = list(map(tokenize, texts))
    model = gensim.models.word2vec.Word2Vec(sentences=token_train_data, size=size)
    return model


def load_embedding_matrix(filepath):
    """Load a pre-trained embedding matrix
    Hint: save and load your embeddings to save time tweaking your model.

    returns:
    :embedding_matrix: a dictionary mapping words to word-vectors (embeddings).
    """
    return gensim.models.word2vec.Word2Vec.load(filepath)


def filter_tokens(tokens, stop):
    """Removes stopwords from a tokens list"""
    return list(filter(lambda x: x not in stop, tokens))


def convert_tokens(tokens, embd, max_seq_length):
    """Converts a list of tokens into a numpy array of fixed shape[1]"""
    if len(tokens) >= max_seq_length:
        return np.vstack(map(lambda x: embd[x], tokens[:max_seq_length]))
    else:
        words = np.vstack(list(map(lambda x: embd[x], tokens)))
        return np.vstack([words, np.zeros((max_seq_length - words.shape[0], words.shape[1]))])


def to_word_vectors(tokenized_samples, embedding_matrix, max_seq_length):
    """Convert the words in each sample into word-vectors.

    params:
    :tokenized_samples: a list of tokenized text samples.
    :embedding_matrix: a dictionary mapping words to word-vectors.
    :max_seq_length: the maximum word-length of the samples.

    returns: a matrix containing the word-vectors of the samples with size:
    (num_samples, max_seq_length, word_vector_size).
    """
    return np.stack([convert_tokens(tokenized_samples[i], embedding_matrix, max_seq_length)
                     for i in range(len(tokenized_samples))])


def generate_batches(data, labels, batch_size, embedding_matrix=None, max_seq_length=140):
    """"Generate batches of data and labels.
    Hint: tokenize

    returns: batch of data and labels. When an embedding_matrix is passed in,
    data is tokenized and returned as matrix of word vectors.
    """
    i = 0
    b = batch_size
    m = len(data)

    while True:
        low = int(b * (i % np.ceil(m / b)))
        high = int(b * (i % np.ceil(m / b)) + b)
        batch_data, batch_labels = data[low:high], labels[low:high]
        if embedding_matrix is not None:
            batch_data = to_word_vectors(batch_data, embedding_matrix, max_seq_length)
        yield batch_data, batch_labels
        i += 1
