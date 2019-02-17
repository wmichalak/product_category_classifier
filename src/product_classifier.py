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

# from keras.preprocessing.image import ImageDataGenerator
# from keras import layers
# from keras import models
# from keras.models import Sequential
# from keras.layers.normalization import BatchNormalization
# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D
# from keras.layers.core import Activation
# from keras.layers.core import Flatten
# from keras.layers.core import Dropout
# from keras.layers.core import Dense
# from keras import backend as K

def read_categories():
    '''
    Read category table. First column are labels, remainder are synonyms
    :return:
    '''
    # Read in categories. First column is desired labels, remainder are synonyms
    categories_raw = pd.read_csv('../data/product_categories.txt', '\t', header = None)

    category_dict = {}
    for row in range(0,categories_raw.shape[0]):
        category_dict[categories_raw.iloc[row,0].lower()] = categories_raw.iloc[row,1:].dropna()

    return category_dict

def acquire_validation_data():

    basedir = os.path.dirname(os.path.abspath(__file__))[:-3]

    # Read URLs and descriptions
    with open('../data/product_data.json', 'r') as f:
        json_text = f.read()

    # Decode into a dict
    catalogue = json.loads(json_text)

    # Read in category data and get synonym dictionary
    category_dict = read_categories()

    # Retrieve test image files from web
    if not os.path.exists('../data/val_data/images/'):
        os.makedirs('../data/val_data/images/')

    # Create table with photo title, identified label, downloaded, description, url
    val_database = pd.DataFrame(columns=['title', 'label', 'downloaded', 'description', 'url'])

    # Iterate over each image in validation set, identify the appropriate label and save image in corresponding folder
    # Aim is to have unknowns fall into the Other category, which will be hand labeled
    for i, item in enumerate(catalogue):
        val_database.loc[i, 'title'] = 'p' + str(i) + '.jpg'
        val_database.loc[i, 'label'] = get_label_validation_data(item['description'], category_dict)
        val_database.loc[i, 'description'] = item['description']
        val_database.loc[i, 'url'] = item['images_url']

        label_dir = '../data/val_data/images/' + val_database.loc[i, 'label']
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # Download the image
        photo_filename = label_dir + '/' + val_database.loc[i, 'title']
        if not os.path.exists(photo_filename):
            try:
                urllib.request.urlretrieve(item['images_url'], photo_filename)
                val_database.loc[i, 'downloaded'] = True
            except:
                print('Cannot download image')
                val_database.loc[i, 'downloaded'] = False
        else:
            val_database.loc[i, 'downloaded'] = True

    val_database.to_csv('../data/val_database.csv')

    return val_database

def get_label_validation_data(description, category_dict):
    # Set up word lemmatizer to make searching in the descriptions more productive

        description = nlp(description)
        sentence = " ".join([token.lemma_ for token in description])

        # Search for category in description and assign, otherwise set to other
        for key in category_dict.keys():
            if key in sentence:
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
    catalogue, successful_download = acquire_validation_data()
    # train_dir, test_dir = acquire_training_data()

    # process the image data
    # train_datagen = ImageDataGenerator(rescale=1/255)
    # test_datagen =  ImageDataGenerator(rescale=1/255)
    #
    # train_generator = train_datagen.flow_from_directory(
    #         train_dir,
    #         target_size=(200, 200),
    #         batch_size=20,
    #         class_mode=None
    #         )
    #
    # test_generator = train_datagen.flow_from_directory(
    #         train_dir,
    #         target_size=(150, 150),
    #         batch_size=20,
    #         class_mode=None
    #         )
    #
    # # Define the CNN
    #
    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (3, 3), activation='relu',
    #                         input_shape=(150, 150, 3)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dense(11, activation='sigmoid'))


