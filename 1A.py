# Aryana Collins Jackson
# R00169199
# Assignment 3 
# Part 1A

# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from numpy import array
from numpy import argmax
np.set_printoptions(threshold=np.nan)

from six.moves import cPickle as pickle

import time

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from datetime import datetime

# -----------------------------------------------------------------------------

# The first function loads the data and separates it into training and testing

def loadData():
        
    pickle_file = 'data.pickle'
    with open(pickle_file, 'rb') as fObj:
        
        imageData = pickle.load(fObj)
        train_X = imageData['train_X']
        train_y = imageData['train_y']
        test_X = imageData['test_X']
        test_y = imageData['test_y']

        print (train_X.shape, train_y.shape)
        print (test_X.shape, test_y.shape)
            
        return train_X, train_y, test_X, test_y

# -----------------------------------------------------------------------------

# Allow users to pick two letters for class 0 and class 1 for the binary 
# regression. A dictionary is created which enumerates the alphabet. The user-
# picked letters are then matched up with the values in the dictionary. Those 
# numbers are returned
      
def pickLetter():
    
    letter0 = input("select letter: ")
    letter1 = input("select letter: ")
                
    # following two lines taken from Dr. Jason Brownlee:
    # https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    
    letterNum0 = char_to_int.get(str(letter0))
    letterNum1 = char_to_int.get(str(letter1))
    
    return letterNum0, letterNum1

# -----------------------------------------------------------------------------

# This code adapted from Dr. Jason Brownlee:
# https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
# The function takes in a subset of the data and one-hot encodes it for the 
# binary regression function

def encoder(subset):
    
    values = array(subset)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    return onehot_encoded

# -----------------------------------------------------------------------------

# This function takes the training and testing data and pushes it through the 
# tensors in order to accurately classify the images of the two letters picked

def binaryLinReg(train_X, train_y, test_X, test_y):
    
    # reset the graph
    tf.reset_default_graph()

    # Parameters
    learning_rate = 0.01
    num_Iterations = 1000
    display_step = 100

#    The tensors containing the images and labels are created. Shapes are 
#    specified, but the number of rows is not set to allow for training and 
#    testing data to be run through the same tensor

    X = tf.placeholder(tf.float32, [None, train_X.shape[1]], name='image')
    Y = tf.placeholder(tf.float32, [None, 2])
    
#    The weights and bias are created as tensors. We want random numbers to 
#    start with. The shape of the weights tensor must be the number of features 
#    by the number of neurons.
    
    w = tf.Variable(tf.random_normal([train_X.shape[1], 2], mean=0.0, stddev=0.05))
    b = tf.Variable([0.])
    
    with tf.name_scope("Model") as scope:
    
        y_pred = tf.matmul(X, w) + b
        
        # here the sigmoid function is called on the predictions
        y_pred_sigmoid = tf.sigmoid(y_pred) 
    
    with tf.name_scope("Error") as scope:

        # cross-entropy error is calculated
        x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=Y)
        # Calculate the mean cross entropy error
        loss = tf.reduce_mean(x_entropy) 
   
    # keep track of the MSE for the plot
    mseSummary = tf.summary.scalar("Mean Squared Error", loss)
    
    with tf.name_scope("GradientDescent") as scope:
        
        # add gradient descent
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        
        # Round the predictions by the logistical unit to classify as 1 or 0
        predictions = tf.round(y_pred_sigmoid)
    
    with tf.name_scope("Predictions") as scope:
        predictions_correct = tf.cast(tf.equal(predictions, Y), tf.float32)
        accuracy = tf.reduce_mean(predictions_correct)
    
    # for Tensorboard
    currentTime = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "TF_Logs1"
    logdir = "{}/run-{}/".format(root_logdir, currentTime)
    
    with tf.Session() as sess:
        
        # write the files to Tensorboard
        summary_writer = tf.summary.FileWriter( logdir, graph=tf.get_default_graph())
        summary_writer.add_graph(sess.graph)
       
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        for i in range(num_Iterations):
            feed_train = {X: train_X, Y: train_y}
            sess.run(train_step, feed_dict=feed_train)
            
            # so the values aren't printed with every iteration
            if i % display_step == 0:
                currentLoss, train_accuracy = sess.run([loss, accuracy], feed_train)
                print('Iteration: ', i,' Loss: ', currentLoss, ' Accuracy: ',train_accuracy)
                _, c, currentMse = sess.run([train_step, loss, mseSummary], feed_dict={X: train_X, Y: train_y})
                summary_writer.add_summary(currentMse, i)
        
                feed_test = {X: test_X, Y: test_y}
                print('Test Accuracy: ', sess.run(accuracy, feed_test))
     
# -----------------------------------------------------------------------------
    
def main():
        
    # load the data
    train_X, train_y, test_X, test_y = loadData()
    
    print ('train x shape 1', train_X.shape)
    
    # pick two letters
    letterNum0, letterNum1 = pickLetter()

    # subset the data, pulling out only rows involving those two letters
    indexTr1 = train_y == letterNum0
    indexTr2 = train_y == letterNum1
    
    allTrIndices = indexTr1 | indexTr2
    
    train_X = train_X[allTrIndices]
    train_y = train_y[allTrIndices]
    
    indexT1 = test_y == letterNum0
    indexT2 = test_y == letterNum1
    
    allTIndices = indexT1 | indexT2
    
    test_X = test_X[allTIndices]
    test_y = test_y[allTIndices]
    
    print ('train x shape 2',train_X.shape)
        
    train_y = train_y.reshape(-1,1)
    test_y = test_y.reshape(-1,1)
    
    # one hot encode train_y and test_y
    train_y = encoder(train_y)
    test_y = encoder(test_y)
    
    # Run the binary linear regression classification
    binaryLinReg(train_X, train_y, test_X, test_y)
    
main()