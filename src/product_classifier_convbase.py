'''
Fashion product category classifier with a pre-trained NN base. In this case we are only training the last dense layer.
'''

import product_classifier
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(200, 200, 3))


# Run the convolutional base over our dataset to get the inputs into the flat layer

batch_size = 20
# Define data directories
train_dir = '../data/train_data/'
test_dir = '../data/test_data/'
val_dir = '../data/val_data/'

train_generator = ImageDataGenerator(rescale=1/255)
test_generator = ImageDataGenerator(rescale=1/255)
validation_generator = ImageDataGenerator(rescale=1/255)

def extract_features(datagen, directory, sample_count):
    """
    Extract features from the convolutional base
    :param datagen: Keras image generator
    :param directory: data directory
    :param sample_count: total number of samples to extract
    :return:
    """

    # Number of features to create for the final hidden layer - this is the output CNNbase and input into the dense layer
    features = np.zeros(shape=(sample_count, 6, 6, 512))
    labels = np.zeros(shape=(sample_count, 10)) # Using 10 for the number of classes
    generator = datagen.flow_from_directory(
        directory,
        color_mode='rgb',
        target_size=(200, 200),
        batch_size=batch_size,
        class_mode='categorical')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size, :] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

train_features, train_labels = extract_features(train_generator, train_dir, 3861)
validation_features, validation_labels = extract_features(validation_generator, val_dir, 763)
test_features, test_labels = extract_features(test_generator, test_dir, 622)



train_features = np.reshape(train_features, (3861, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (763, 4 * 4 * 512))
test_features = np.reshape(test_features, (622, 4 * 4 * 512))


# Define the Keras densely connected classifier

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

product_classifier.create_plots(model, 'accuracy_base.png', 'loss_base.png')
