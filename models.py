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

# This module contains a class template for building machine learning models
# on the IMDB movie review dataset. The code is heavily commented to allow you to follow along easily.

# Please report any bugs you find using @yazabi, or email us at contact@yazabi.com.
from __future__ import print_function
from __future__ import generators
from __future__ import division

# import tensorflow as tf     # (optional) feel free to build your models using keras

import data_preprocessing as dp


## recommended for LSATextClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

## recommended for LSTMTextClassifier
from keras.models import Sequential
from keras.layers import Conv1D, Dense, GlobalMaxPooling1D, Dropout, BatchNormalization, LSTM


class ClassifierTemplate(object):
    """Fill out this template to create three classes:
    LSATextClassifier(object)
    CNNTextClassifier(object)
    LSTMTextClassifier(object)

    Modify the code as much as you need.
    Add arguments to the functions and add as many other functions/classes as you wish.
    """

    def __init__(self, embedding_matrix=None):
        """Initialize the classifier with an (optional) embedding_matrix
        and/or any other parameters."""
        self.embedding_matrix = embedding_matrix
        self._build()

    def _build(self, model_parameters=None):
        """Build the model/graph."""
        raise NotImplementedError

    def train(self, train_data, train_labels, batch_size=50, num_epochs=5):
        """Train the model on the training data."""
        raise NotImplementedError

    def evaluate(self, test_data, test_labels):
        """Evaluate the model on the test data.

        returns:
        :accuracy: the model's accuracy classifying the test data.
        """
        raise NotImplementedError

    def predict(self, review):
        """Predict the sentiment of an unlabelled review.

        returns: the predicted label of :review:
        """
        raise NotImplementedError


class LSATextClassifier(ClassifierTemplate):
    """Holds a Latent Semantic Analysis classifier with a logistic regression model"""
    def __init__(self, embedding_matrix=None):
        super().__init__(embedding_matrix)

    def _build(self, model_parameters=None):
        """Build the model."""
        self.vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=2, use_idf=True, stop_words='english')
        self.lsa = make_pipeline(TruncatedSVD(100), Normalizer(copy=False))
        self.model = LogisticRegression()

    def train(self, train_data, train_labels, batch_size=50, num_epochs=5, additional_parameters=None):
        """Train the model"""
        train_tfidf = self.vectorizer.fit_transform(train_data)
        train_lsa = self.lsa.fit_transform(train_tfidf)
        self.model.fit(train_lsa, train_labels)

    def evaluate(self, test_data, test_labels, additional_parameters=None):
        test_tfidf = self.vectorizer.transform(test_data)
        test_lsa = self.lsa.transform(test_tfidf)
        return self.model.score(test_lsa, test_labels)

    def predict(self, review):
        review_tfidf = self.vectorizer.transform([review])
        review_lsa = self.lsa.transform(review_tfidf)

        return self.model.predict(review_lsa)


class CNNTextClassifier(ClassifierTemplate):
    """Holds a CNN classifier"""

    def __init__(self, embedding_matrix=None):
        super().__init__(embedding_matrix)

    def _build(self, model_parameters=None):
        self.model = Sequential()
        self.model.add(Conv1D(filters=256, kernel_size=3, padding='valid', input_shape=(140, 100)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(Conv1D(filters=256, kernel_size=3, padding='valid'))
        self.model.add(BatchNormalization())

        self.model.add(GlobalMaxPooling1D())

        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, train_data, train_labels, batch_size=100, num_epochs=2):
        data_generator = dp.generate_batches(train_data, train_labels, batch_size, self.embedding_matrix, 140)

        self.model.fit_generator(data_generator, len(train_data) // batch_size, epochs=num_epochs)

    def evaluate(self, test_data, test_labels, batch_size=100):
        data_generator = dp.generate_batches(test_data, test_labels, batch_size, self.embedding_matrix, 140)

        return self.model.evaluate_generator(data_generator, len(test_data) // batch_size)

    def predict(self, review):
        tokens = dp.process_data([review])
        vectorized = dp.to_word_vectors(tokens, self.embedding_matrix, 140)

        return self.model.predict(vectorized)


class RNNTextClassifier(ClassifierTemplate):
    """Holds an LSTM classifier"""
    def __init__(self, embedding_matrix=None):
        super().__init__(embedding_matrix)

    def _build(self, model_parameters=None):
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(140, 100), return_sequences=True))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(.2))

        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(.2))

        self.model.add(LSTM(32))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(.2))

        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, train_data, train_labels, batch_size=100, num_epochs=2):
        data_generator = dp.generate_batches(train_data, train_labels, batch_size, self.embedding_matrix, 140)

        self.model.fit_generator(data_generator, len(train_data) // batch_size, epochs=num_epochs)

    def evaluate(self, test_data, test_labels, batch_size=100):
        data_generator = dp.generate_batches(test_data, test_labels, batch_size, self.embedding_matrix, 140)

        return self.model.evaluate_generator(data_generator, len(test_data) // batch_size)

    def predict(self, review):
        tokens = dp.process_data([review])
        vectorized = dp.to_word_vectors(tokens, self.embedding_matrix, 140)

        return self.model.predict(vectorized)
