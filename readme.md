
### Introduction ###
The purpose of this project is to classify the category of clothing observed in an image into one of the following categories:
Dresses, Tops, Jeans, Skirts, Rompers, Shoes, Bags, Jewelry, Swimwear, Intimates, Others.

For this project I will used a multilabel CNN classifier based in the Keras API.

The project consists of X parts:
1. Acquire the validation data. This was provided to me in a JSON file with urls and descriptions of the images. 
2. Acquire a train / test dataset to learn coefficients of the CNN
3. Perform data preprocessing to 
3. Define the CNN structure
4. Adjust the CNN structure


### Getting the training data ###

To train the model, I needed labeled training data. I decided to download labeled training data from Google images by searching
for the categories that I am trying to classify and download those images.  I used the 
[google images download project](https://github.com/hardikvasa/google-images-download). In order to acquire more than 100 images for each category,
I had to download the Chrome webdriver that is used with selenium and modify line 853 in googe_images_download with the path
to the chromedriver executable. 


I used the description assigned to each picture to generate the label. In order to use the description I
1. Broke the sentence down into parts
2. Lemmatized

considered accessing synonyms words straight from the thesaurus, but decided that there were too many overlaps accross the 
categories and decided to generate my own dictionary