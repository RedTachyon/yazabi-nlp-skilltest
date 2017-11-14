#!/bin/python3

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

# This script contains a loose template for building and training machine learning models
# on the imdb movie review dataset

# Please report any bugs you find using @yazabi, or email us at contact@yazabi.com.
from __future__ import print_function
from __future__ import generators
from __future__ import division

import sys
import data_preprocessing as dp
from models import LSATextClassifier
from models import CNNTextClassifier
from models import RNNTextClassifier

LEARNING_RATE = 1e-4
BATCH_SIZE = 100
NUM_EPOCHS = 2
MAX_SEQ_LENGTH = 140
N_FEATURES = 200
EMBEDDING_SIZE = 100
FILENAME = 'embeddings'

if __name__ == "__main__":

    use_model = sys.argv[1]
    if use_model is None:
        print('Specify model to be used. LSATextClassifier, RNNTextClassifier or CNNTextClassifier')
        sys.exit()

    # load data
    print("Loading data (this might take ~4 minutes)")
    train_data, test_data, train_labels, test_labels = dp.load_imdb_data()

    # build and train model
    if use_model == 'LSATextClassifier':
        train_data, test_data = dp.clean_data(train_data), dp.clean_data(test_data)
        model = LSATextClassifier()
        model.train(train_data, train_labels)
        accuracy = model.evaluate(test_data, test_labels)

    elif use_model == "TextRNN":
        print("Generating the embedding matrix")
        embd_matrix = dp.make_embedding_matrix(train_data + test_data, size=EMBEDDING_SIZE)
        # embd_matrix = dp.load_embedding_matrix(FILENAME)
        train_tokens, test_tokens = dp.process_data(train_data), dp.process_data(test_data)

        model = RNNTextClassifier(embd_matrix)
        model.train(train_tokens, train_labels, BATCH_SIZE, NUM_EPOCHS)
        loss, accuracy = model.evaluate(test_tokens, test_labels)

    elif use_model == 'TextCNN':
        print("Generating the embedding matrix")
        embd_matrix = dp.make_embedding_matrix(train_data + test_data, size=EMBEDDING_SIZE)
        # embd_matrix = dp.load_embedding_matrix(FILENAME)
        train_tokens, test_tokens = dp.process_data(train_data), dp.process_data(test_data)

        model = CNNTextClassifier(embd_matrix)
        model.train(train_tokens, train_labels, num_epochs=1)
        loss, accuracy = model.evaluate(test_tokens, test_labels)


    else:
        model = None
        raise ValueError("The model should be one of LSATextClassifier, TextRNN or TextCNN")

    # evaluate model
    print('Test accuracy: ', accuracy)

    # predict
    neg_review = 'This movie was the worst thing I have ever watched.'
    pos_review = 'This was the greatest thing. I really liked it.'
    neg_pred = model.predict(neg_review)
    pos_pred = model.predict(pos_review)
    print('Prediction on negative review:', neg_pred)
    print('Prediction on positive review:', pos_pred)

    custom_review = input("Enter your own review:\n")
    print('Prediction on your review:', model.predict(custom_review))

