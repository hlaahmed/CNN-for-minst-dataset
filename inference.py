#!/usr/bin/env python
# coding: utf-8
import numpy as n
from keras import models
import cv2
import matplotlib.pyplot as p

# in this function we pass as a parameter the path for a testing image, the output should be the correct label for this image
image_path = "img_345.jpg"
def predict(image_path):
    # we read the image from the specified path
    im = cv2.imread(image_path)
    #then we load the model that we previosly saved in model.h5
    model = models.load_model("model.h5")
    # we expand the dimensions of our input image, so that it suits the input size of our model
    expand = n.expand_dims(im, axis=0)
    # we call the predict function of our model, and give it the input image after expansion as a parameter
    # and the function returns the correct label for the image
    digit = n.argmax(model.predict(expand), axis=-1)
    p.imshow(im)
    p.show()
    print(" the predicted label is:")
    print(digit[0])
predict(image_path)





