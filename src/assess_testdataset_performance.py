'''
Test performance of fashion product category classifier using test set
'''

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
from keras.models import load_model


model = load_model('fashion_classifier_1.h5')





