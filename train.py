#!/usr/bin/env python
# coding: utf-8
import numpy as n
import os
import matplotlib.pyplot as p
from sklearn.model_selection import train_test_split
import cv2
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
directorydata = "D:/trainingSet"       #here we write the path to our training set
categories = ["0","1","2","3","4","5","6","7","8","9"] #the different categories in minst dataset

# in this function we are going to load the data from minst dataset and return two arrays, one of them contains the images, and the other contains the lables for these images
# the function takes the directory of the dataset as a parameter
def Load_data(directory):
    training_data = []
    images = []
    labels = []
    for c in categories:
        #here we join the directory path with the folder name
        path = os.path.join(directory, c)
        #different folders correspond to different classes, and we need the label to be a number not a character
        #so we create an array of labels and store in it the index for each class, so it contains numbers from 0 to 9
        label = categories.index(c)
        for img in os.listdir(path):
            try:
                    # here we are going to fill our training_data array with the images and their labels, by reading each image in
                    # the specified path and adding it and its label to training data array
                    img_array = cv2.imread(os.path.join(path,img))
                    training_data.append([img_array, label]) 
            except Exception as ex: #incase of unfound image we pass
                pass
    import random
    # we are going to mix the array of training_data in a random way, so that labels and images are no longer in a sequential order
    random.shuffle(training_data)
    #after that we separate the images and the labels into two different arrays
    for features, label in training_data:
        images.append(features)
        labels.append(label)
    # our data size is 3x28x28 since they are RGB images, they have 3 channels
    images = n.array(images).reshape(-1,28,28, 3) 
    return(images,labels)


# in the following function we are going to do some preprocessing, creation and training of our model, save model.h5 and return the model
# also after training we are going to choose a sample of training images and compare between their predicted and real labels
# to make sure our model is working
def train(images,labels):
    # here we are going to split images and labels, so that we use most of them for training, and the rest are used for validation
    # so that we avoid overfitting of our model
    train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.25, shuffle=True, random_state=30)
    print(train_x.shape)
    print(test_x.shape)
    #we convert our labels to hot_encode vector, so that it gives value 1 at the correct label and zero for the rest
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    # we change the type of the values of images to float to perform normalization, and make values range from zero to one
    #this help us to converge faster in the training
    train_norm = train_x.astype('float32')
    test_norm = test_x.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    #we start the creation of our model
    model = Sequential() # sequantial means a linear stack of layers
    # we add a convolution layer, we choose to have 32 filters, each of size 3x3, the activation we chose is relu 
    # we choose to initialize the kernel filter weights with 'he_uniform'
    # and we know that the size of our images is 3x28x28, so we enter it as the input_shape
    # the stride in convolution layer is set by default to 1
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 3)))#inputsize:3x28x28 outputsize:32x26x26
    # the next layer is maxpooling layer and we use it to reduce the dimensions of the output
    # we choose pool size to be 2x2 and we didn't specify the stride, so the default is that it will be the same as pool size 
    model.add(MaxPooling2D((2, 2)))#inputsize:32x26x26 outputsize:32x13x13
    # we have a flatten layer that collapses dimensions into channel dimension
    model.add(Flatten())#inputsize:32x13x13 outputsize:5408
    # we add dense wich is a fully connected layer, we specified the output size of the layer to be 100, and again we used relu and he_unifrom
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))#inputsize:5408 outputsize:100
    # we add another fully connected layer, we specified the output size of the layer to be 10, but here we used softmax as activation instead of relu 
    model.add(Dense(10, activation='softmax'))#inputsize:100 outputsize:10
    # we chose stochastic gradient descent
    opt = SGD(lr=0.01, momentum=0.9)
    # we are going to see during training the model, the loss and accuracy of the model, and how they change as the training advance
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # we train and validate the model in 20 iterations
    model.fit(train_norm, train_y, epochs=20, batch_size=128, validation_data=(test_norm,test_y))
    model.save("model.h5")
    model.summary()
    # here we choose a sample of 4 training images and compare between their predicted and real labels
    # to make sure our model is working
    predict = model.predict_classes(test_norm[:4]) 
    print(predict, test_y[:4])
    for i in range(4):
        p.imshow(test_norm[i])
        p.show()
    return model
images,labels = Load_data(directorydata)
train(images,labels)

