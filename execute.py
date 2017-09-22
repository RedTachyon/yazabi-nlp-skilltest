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
# from models import LSATextClassifier
# from models import CNNTextClassifier
# from models import RNNTextClassifier

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 10
MAX_SEQ_LENGTH = 1000
N_FEATURES = 200

if __name__ == "__main__":

    use_model = sys.argv[1]
    if use_model is None:
        print('Specify model to be used. LSATextClassifier, RNNTextClassifier or CNNTextClassifier')
        sys.exit()

    # load data
    X_train, X_test, y_train, y_test = dp.load_imdb_data()

    # build and train model
    if use_model == 'LSATextClassifier':
        model = LSATextClassifier()
        model.build()
        model.train(X_train, y_train)

    elif use_model == "TextRNN":
        embd_matrix = dp.make_embedding_matrix(X_train + X_test, size=EMBEDDING_SIZE)
        # embd_matrix = dp.load_embedding_matrix(FILENAME)
        model = RNNTextClassifier()
        model.build()
        model.train(X_train, y_train, BATCH_SIZE, NUM_EPOCHS)

    elif use_model == 'TextCNN':
        embd_matrix = dp.make_embedding_matrix(X_train + X_test, size=EMBEDDING_SIZE)
        # embd_matrix = dp.load_embedding_matrix(FILENAME)
        model = CNNTextClassifier()
        model.build()
        model.train(X_train, y_train, BATCH_SIZE, NUM_EPOCHS)

    else:
        model = None
        raise ValueError("The model should be one of LSATextClassifier, TextRNN or TextCNN")

    # evaluate model
    accuracy = model.evaluate(X_test, y_test)
    print('Test accuracy: ', accuracy)

    # predict
    neg_review = 'This movie was the worst thing I have ever watched.'
    pos_review = 'This was the greatest thing. I really liked it.'
    neg_pred = model.predict(neg_review)
    pos_pred = model.predict(pos_review)
    print('Prediction on negative review:', neg_pred)
    print('Prediction on positive review:', pos_pred)