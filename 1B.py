# Aryana Collins Jackson
# R00169199
# Assignment 3 
# Part 1B

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

def softmaxClassification(train_X, train_y, test_X, test_y):
    
    # reset the graph
    tf.reset_default_graph()

    # Parameters
    learning_rate = 0.03
    num_Iterations = 1000
    display_step = 100
    
    with tf.name_scope("Data") as scope:

    #    The tensors containing the images and labels are created. Shapes are 
    #    specified, but the number of rows is not set to allow for training and 
    #    testing data to be run through the same tensor
        
        X = tf.placeholder(tf.float32, [None, 784], name='image')
        Y = tf.placeholder(tf.int32, [None, 10], name='label')
    
    with tf.name_scope("WeightsBias") as scope:
        
    #    The weights and bias are created as tensors. We want random numbers to 
    #    start with. The shape of the weights tensor must be the number of 
    #    features by the number of neurons.

        w = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=0.05))
        b = tf.Variable(tf.zeros([1,10]))
    
    with tf.name_scope("Model") as scope:
        
    #    The logits node completes the regression by multiplying the weights by 
    #    the features and then adding the bias
    
        logits = tf.matmul(X, w) + b
    
    with tf.name_scope("Error") as scope:
        
#        This is where the error is calculated. It is run through the softmax 
#        cross entropy function which takes in the predictions as well as the 
#        actual data to calculate the total error
    
        error = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=Y, name='loss')    
        loss = tf.reduce_mean(error) 

#    We create an optimizer that does gradient descent in order to find the 
#    minimum cost
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
    
#    We apply a softmax function that returns the 10 softmax values for each 
#    training example. It's in the form of a matrix (training examples by number 
#    of neurons). This will be a matrix in. 'pred' is a matrix containing the 
#    logits for each of the ten neurons for each training example
    
    preds = tf.nn.softmax(logits)
    
#    tf.argmax takes preds as an argument and returns the column index that has 
#    the largest value. This corresponds to the class that SoftMax predicts as 
#    being most probable. tf.equal compares each of the predicted values with 
#    the actual labels and returns a tensor of booleans 
    
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    
#    correct_preds is passed through and we turn it into float values (Trues 
#    become 1, Falses become 0. The total is summed. 
    
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    
    # keep track of the accuracy for plots
    accuracySummary = tf.summary.scalar("Accuracy", accuracy)

#    Set the directory and time stamp for the Tensorboard graph
    currentTime = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "TF_Logs"
    logdir = "{}/run-{}/".format(root_logdir, currentTime)    

    # Start training
    with tf.Session() as sess:
        
        # write the files to Tensorboard
        summary_writer = tf.summary.FileWriter( logdir, graph=tf.get_default_graph())
        summary_writer.add_graph(sess.graph)
        
#        Keep track of how long the program takes to run. Then initialize all 
#        the variables
        start_time = time.time()
        sess.run(tf.global_variables_initializer())
  
        for i in range(num_Iterations):
            _, lossIteration = sess.run([optimizer, loss], {X: train_X, Y:train_y})
            final_accuracy = sess.run(accuracy, {X: test_X, Y:test_y})
            
            # so the values aren't printed with every iteration
            if i % display_step == 0:
                currentLoss, train_accuracy = sess.run([loss, accuracy], {X: train_X, Y: train_y})
                print('Iteration: ', i,' Loss: ', lossIteration, ' Accuracy: ',final_accuracy/test_X.shape[0])
               
                _, c, currentAcc = sess.run([optimizer, loss, accuracySummary], feed_dict={X: train_X, Y: train_y})
                summary_writer.add_summary(currentAcc, i)
            
#                print('Total time: {0} seconds'.format(time.time() - start_time))
   
# -----------------------------------------------------------------------------
    
def main():
        
    # load the data
    train_X, train_y, test_X, test_y = loadData()
    
#    print ('train x shape 1', train_X.shape)
    
    # one hot encode train_y and test_y
    train_y = encoder(train_y)
    test_y = encoder(test_y)
    
    # Run the classification function
    softmaxClassification(train_X, train_y, test_X, test_y)
    
main()