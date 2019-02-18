'''
Fashion product category classifier
'''

import json
import urllib
import os
from google_images_download import google_images_download
import pandas as pd
import shutil
import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers

# Issues with OpenMP force me to set this variable in order to run on my Mac OsX machine
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def read_categories():
    '''
    Read category table. First column are labels, remainder are synonyms
    :return: category_dict
    '''
    # Read in categories. First column is desired labels, remainder are synonyms
    categories_raw = pd.read_csv('../data/product_categories.txt', '\t', header = None)

    category_dict = {}
    for row in range(0,categories_raw.shape[0]):
        category_dict[categories_raw.iloc[row,0].lower()] = categories_raw.iloc[row,1:].dropna()

    return category_dict

def acquire_test_data():
    '''
    Download and label the validation dataset from the product_data.json file supplied with the project
    :return: val_database pandas DataFrame
    '''

    basedir = os.path.dirname(os.path.abspath(__file__))[:-3]

    # Read URLs and descriptions
    with open('../data/product_data.json', 'r') as f:
        json_text = f.read()

    # Decode into a dict
    catalogue = json.loads(json_text)

    # Read in category data and get synonym dictionary
    category_dict = read_categories()

    # Retrieve test image files from web
    if not os.path.exists('../data/test_data/'):
        os.makedirs('../data/test_data/')

    # Create table with photo title, identified label, downloaded, description, url
    test_database = pd.DataFrame(columns=['title', 'label', 'downloaded', 'description', 'url'])

    # Iterate over each image in validation set, identify the appropriate label and save image in corresponding folder
    # Aim is to have unknowns fall into the Other category, which will be hand labeled

    cannot_download = 0
    for i, item in enumerate(catalogue):
        test_database.loc[i, 'title'] = 'p' + str(i) + '.jpg'
        test_database.loc[i, 'label'] = get_labels_on_test_data(item['description'], category_dict)
        test_database.loc[i, 'description'] = item['description']
        test_database.loc[i, 'url'] = item['images_url']

        label_dir = '../data/test_data/' + test_database.loc[i, 'label']
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # Download the image
        photo_filename = label_dir + '/' + test_database.loc[i, 'title']
        if not os.path.exists(photo_filename):
            try:
                urllib.request.urlretrieve(item['images_url'], photo_filename)
                test_database.loc[i, 'downloaded'] = True
            except:
                cannot_download += 1
                print('Cannot download image:' + test_database.loc[i, 'title'] + ', total={}'.format(cannot_download))
                test_database.loc[i, 'downloaded'] = False
                test_database.loc[i, 'label'] = 'nan'
        else:
            test_database.loc[i, 'downloaded'] = True

    test_database.to_csv('../test_database.csv')

    # Report totals
    categories_dummy = pd.get_dummies(test_database['label'])
    for col  in categories_dummy.columns:
        print(col, categories_dummy[col].sum())

    return test_database

def get_labels_on_test_data(description, category_dict):
    ''' Identify label from the image description. Used for unlabeled validation dataset. Use spacy natural language processor.
    :param description: item description as a string
    :param category_dict: dict of labels and keywords
    :return: identified label
    '''

    description = nlp(description)
    sentence = " ".join([token.lemma_ for token in description])

    # Search for category in description and assign, otherwise set to other
    for key in category_dict.keys():

        if key == 'jewelry':
            hold = 1

        if key in sentence:

            # Forced Labeling rules for superceding
            if key == 'top' and 'bikini' in sentence:
                return 'swimwear'
            else:
                return key
        else:
            for syn in category_dict[key]:
                if syn in sentence:
                    return key

    return 'other'

def acquire_training_data():
    '''Scrape training data from the web'''

    basedir = os.path.dirname(os.path.abspath(__file__))[:-3]

    # Get the keywords of interest
    with open('../data/product_categories.txt') as f:
        categories = f.readlines()

    # Remove others for getting data
    categories.remove('Others')

    # Create string format for download method
    keywords = ','.join(list(map(lambda s: s.strip('\n'), categories)))

    # Gather training data from the web - using 'google_images_download' by:
    # https: // github.com / hardikvasa / google - images - download

    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": keywords, "limit": 500,
                 "print_urls": False}  # creating list of arguments
    paths = response.download(arguments)

    # Move data
    for category in categories:
        category = category.strip('\n')
        # Make dirs
        train_dir = basedir + 'data/train_data/' + category
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        test_dir = basedir + 'data/test_data/' + category
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        # Copy 80/20 split to train/test
        files = os.listdir('downloads/' + category)
        for i in range(0, int(len(files) * .8)):
            shutil.copy(basedir + 'src/downloads/' + category + '/' + files[i],
                        train_dir + '/' + files[i])
        for i in range(int(len(files) * .8) + 1,len(files)):
            shutil.copy(basedir + 'src/downloads/' + category  + '/' + files[i],
                        test_dir + '/' + files[i])

    return train_dir, test_dir

if __name__ == '__main__':

    # Acquiring data, comment out after use
    if os.path.exists('../test_database.csv'):
        test_database = pd.read_csv('../test_database.csv')
    else:
        test_database = acquire_test_data()

    # Acquire the training data # Commented out after run
    # train_dir, test_dir = acquire_training_data()

    # process the image data
    train_datagen = ImageDataGenerator(rescale=1/255)
    test_datagen = ImageDataGenerator(rescale=1/255)
    val_datagen = ImageDataGenerator(rescale=1/255)

    train_dir = '../data/train_data/'
    test_dir = '../data/test_data/'
    val_dir = '../data/val_data/'

    print('Loading train data...')
    train_generator = train_datagen.flow_from_directory(
            train_dir,
            color_mode='rgb',
            target_size=(200, 200),
            batch_size=20,
            class_mode='categorical'
            )

    print('Loading test data...')
    test_generator = train_datagen.flow_from_directory(
            test_dir,
            color_mode='rgb',
            target_size=(200, 200),
            batch_size=20,
            class_mode='categorical'
            )

    print('Loading validation data...')
    validation_generator = train_datagen.flow_from_directory(
            val_dir,
            color_mode='rgb',
            target_size=(200, 200),
            batch_size=1,
            class_mode='categorical'
            )

    if not os.path.exists('fashion_classifier_1.h5'):
        # Define the CNN
        #
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=(200, 200, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['acc'])

        history = model.fit_generator(
                      train_generator,
                      steps_per_epoch=100,
                      epochs=50,
                      validation_data=validation_generator,
                      validation_steps=50)

        model.save('fashion_classifier_1.h5')

        # Create plots
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig('../data/accuracy.png')

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig('../data/loss.png')
        plt.show()

    else:

        model = models.load_model('fashion_classifier_1.h5')
        history = model.fit_generator(
              train_generator,
              steps_per_epoch=100,
              epochs=20,
              validation_data=validation_generator,
              validation_steps=50)

        model.save('fashion_classifier_2.h5')

        # Create plots
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig('../data/accuracy.png')

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig('../data/loss.png')
        plt.show()

        test_loss, test_acc = model.evaluate_generator(test_generator, steps = 50)
        print('test acc:', test_acc)