
### Introduction ###
The purpose of this project is to classify the category of clothing shown in an image as:
Dresses, Tops, Jeans, Skirts, Rompers, Shoes, Bags, Jewelry, Swimwear, Intimates, Others.

For this project I will used a multilabel CNN classifier using the Keras API.

The project consists of 8 parts:
1. Acquiring the test data and assigning labels based on the description.  
2. Acquire a train / validation dataset to learn coefficients of the CNN.
3. Define image pre-preprocessing
4. Define the CNN structure
5. Train the CNN
6. Assessing performance of the model
7. Perform hyperparameter tuning
8. Perform final validation error assessment

#### 1: Acquiring the test data and assigning labels based on the description.  ####
I used the fashion items given in product_data.json as the test set. I choose to use the given dataset as a test
 dataset for two reasons: 
 
 1. the final test of the model is based on how well the model performs on the provided set and 
 2. the set has imbalanced classes and therefore training the model would have a difficult for the minority classes. 

The first challenge I had to deal with in this project was assigning labels to the images. There are a number of ways I could have 
handled this: generating a theasarus based on the classification categories or using a language analysis toolkit to name two. 
Given the amount of time I wanted to spend on the project, I chose the former approach:
To assign labels to each photo, I created a list of synonyms corresponding each category item and saved this in the categories.txt file.
I considered using the `thesaurus` package to automatically generate a list of synonyms, but found too much overlap across items;
not to mention that removing the non-fashion synonyms would be tricky, e.g. _top_ could mean shirt, maximum, excellent, etc

I ended up with a set of synonyms:

| Dress  | Top        | Jean  | Skirt  | Romper   | Shoe      | Bag      | Jewelry  | Swimwear | Intimates | Others |
|--------|------------|-------|--------|----------|-----------|----------|----------|----------|-----------|--------|
| gown   | blouse     | denim | mini   | jumpsuit | boot      | purse    | necklace | swim     | bra       | pant   |
| frock  | tunic      |       | tutu   |          | sandal    | duffel   | ring     | coverup  | panties   | short  |
| muumuu | shirt      |       | hoop   |          | clog      | pannier  | bracelet | bodysuit | lingerie  |        |
| drape  | tank       |       | midi   |          | heel      | backpack | jewel    | robe     | under     |        |
| smock  | sweatshirt |       | sarong |          | pump      | clutch   | earring  | bikini   |           |        |
|        | turtleneck |       | kilt   |          | slipper   | handbag  | brooch   |          |           |        |
|        | camise     |       | dirndl |          | slip      | tote     | chain    |          |           |        |
|        | camisole   |       |        |          | flip-flop | satchel  | choker   |          |           |        |
|        | polo       |       |        |          | wedge     | pouch    | charm    |          |           |        |
|        | vest       |       |        |          | loafer    |          | diamond  |          |           |        |
|        | blazer     |       |        |          | mary jane |          | ornament |          |           |        |
|        | jacket     |       |        |          | oxford    |          | pendant  |          |           |        |
|        | sleeve     |       |        |          | stilleto  |          | gem      |          |           |        |
|        |            |       |        |          | peep      |          | silver   |          |           |        |
|        |            |       |        |          | platform  |          | gold     |          |           |        |
|        |            |       |        |          | scarpin   |          | platinum |          |           |        |
|        |            |       |        |          | mule      |          |          |          |           |        |
|        |            |       |        |          | trainer   |          |          |          |           |        |
|        |            |       |        |          | cleats    |          |          |          |           |        |
|        |            |       |        |          | footwear  |          |          |          |           |        |


I chose to also implement a few labeling rules to circumvent issues. One major labelling problem was with the word __top__. Too
often, top was used with bikini, for example. Thus, I set a rules for 'bikini' to supercede top. Another conscious decision I made in this 
project was to ignore the __other__ category. Any image that did not fall into one of the first 10 classes was not used.

To assign the labels, I broke down each `description` from product_data.json by tokenizing and lemmatizing and searching 
for the category keywords in the description. I used the `spacy` natural language processor `nlp` for this task.

```
description = nlp(description)
sentence = " ".join([token.lemma_ for token in description])

for key in category_dict.keys():
    if key in sentence:
        return key
    else:
        for syn in category_dict[key]:
            if syn in sentence:
                return key
```

After determining the label, each photo was downloaded using the `urllib` package. 

```
urllib.request.urlretrieve(item['images_url'], photo_filename)
```

I saved each photo in a folder designated by the label. All pictures that did not fall into one of the main categories were labelled as other. 
The other folder was removed from the test data directory after processing based on my decision to not develop a model that 
handled this.
        
A couple of odds and ends to point out:
* In general, there are weaknesses in my approach, e.g., when more than one keyword is present, the algorithm selects the first
one it checks. This is not something I am choosing to spend time on. Rather I am manually fixing the mis-labelled items.
* Since _pants_ are not a unique category (rather, jeans is used), I assign pants to the other category as __jeans__ do not necessarily
cover pants.
* some descriptions were just too vague and manual labelling was required. For example, the description:
_Shop On Top at Urban Outfitters. We have all the latest styles and fashion trends you're looking for right here._
* There were also images in this set that were not fashion items; these images were assigned to 
the __other__ category.
* Some images could not be downloaded and thus were left out of the test set and are designated by the __nan__ category below.
* A summary report of the filename, assigned label, whether the download was complete, the description and url were saved in a file called
val_database.csv (which is in the repository).

In summary, out of 1000 records, the samples spread across the categories:

| Category  | #   |
|-----------|-----|
| bag       | 31  |
| dress     | 73  |
| intimates | 16  |
| jean      | 27  |
| jewelry   | 49  |
| nan       | 213 |
| other     | 166 |
| romper    | 5   |
| shoe      | 44  |
| skirt     | 11  |
| swimwear  | 82  |
| top       | 283 |

All of the images are shown in this montage

![Figure 0: Test data](https://github.com/wmichalak/product_category_classifier/raw/master/data/montage.png)
Figure 0: Montage of test images

While the project goal was to classify all 1000 samples, I clearly cannot given that 213 files could not be downloaded
and 166 of the images did not fall into one of the given categories. If I were building an extensive model I would not be 
limiting myself to these 10 categories and this problem would go away. 

#### 2: Acquiring the training and validation data and assigning labels based on the search criteria.  ####
To train the model, I decided not to use the provided set because  it is relatively small, and the classes are imbalanced. Instead,
I wanted to use another dataset. I came up with two proposals: 1) use an available database on the web or 2) download labeled 
training data from Google images by searching for the categories and downloading those images. Given that the  available 
fashion databases do not have the same classes (despite having lots of labeled images), I decided to go with option 2. To download
images from Google, I used the available package
[google images download project](https://github.com/hardikvasa/google-images-download). 
In order to acquire more than 100 images for each category,
I had to download the Chrome webdriver that is used with selenium and modify line 853 in googe_images_download with the path
to the chromedriver executable. 

```markdown
response = google_images_download.googleimagesdownload()
arguments = {"keywords": keywords, "limit": 500,
             "print_urls": False}  # creating list of arguments
paths = response.download(arguments)
```

All of the data was downloaded to folders with corresponding labels all ready to go for my Keras model.

Some of the files that were downloaded were corrupt and needed to be be removed from the directories.
I walked through the directories and checked (check_images.py) that each image could be opened using matplotlib's image package.

```
train_files = os.walk('../data/train_data', topdown=True)

for root, dirs, files in train_files:
    for name in files:
        if '.jpeg' in name or '.tif' in name or '.jpg' in name or '.gif' in name or '.png' in name:
            try:
                mpimg.imread(os.path.join(root, name))
            except:
                print('Removing: ' + os.path.join(root, name))
                os.remove(os.path.join(root, name))
        else:
            print('Skipped: ' + os.path.join(root, name)) 

```

#### 3. Define image preprocessing: ImageGenerator.flowfromdirectory() with Keras ###

To load and process the images, I used the Keras ImageGenerator suite of tools, which handles the import, scaling, decoding jpeg
into RGB grid of pixels, coversion to floating-point tensors, and rescaling the 0 to 255 to the [0, 1] interval. I load in the 
train, test, and validation sets using:

```

train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)
val_datagen = ImageDataGenerator(rescale=1/255)
    
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

```

While running, Keras confirms the number of images and number of identified classes; which are based on the subfolders in 
the directories.

```
Loading train data...
Found 3861 images belonging to 10 classes.
Loading test data...
Found 622 images belonging to 10 classes.
Loading validation data...
Found 763 images belonging to 10 classes.
```

#### 4. Define the CNN structure ####

As an initial attempt, I used a common CNN structure with 3 layers of 2D convolutions with relu activations 
and pooling, followed by a flattening to a dense layer and a sigmoud activation function for the final activation. 

```
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(200, 200, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='sigmoud'))
```

We can take look at how the dimensions of the feature maps change with each layer:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 198, 198, 32)      896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 99, 99, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 97, 97, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 48, 48, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 46, 46, 128)       73856     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 23, 23, 128)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 21, 21, 128)       147584    
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 10, 10, 128)       0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 12800)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               6554112   
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130      
=================================================================
Total params: 6,800,074
Trainable params: 6,800,074
Non-trainable params: 0
_________________________________________________________________
```

The current model has a total of 6,800,074 parameters.

Lastly, I define the model loss, optimizer and metric using the Keras one-liner:

```
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

```

I use a binary_crossentropy to measure the distance between two probability distributions: the probability distribution
output by the network and the true distribution of the labels. I use accuracy as my metric since the classes are fairly evenly
distributed.

#### 5. Train the CNN ####

To train the model, I use the Keras `fit_generator` method since I am using the `ImageGenerator` function. I will take 100 
steps per epoch, train over 30 epochs, and test the validation data in steps =  the number of images divided by the batchsise.
 Since I am using a generator, I couldconceivable cycle over the images endlessly. Instead, I specify to stop at 763//25 = 30 steps.

```
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=763/50)
      
```

This model took about 40 minutes to train on my Macbook Pro.

#### 6. Assessing performance of the model on the training and validation set ####

A plot of the training and validation accuracy is shown in Figure 1. A plot of the training and
validation loss is shown in Figure 2.


![Figure 1: Training and validation accuracy](https://github.com/wmichalak/product_category_classifier/raw/master/data/1-accuracy.png)
Figure 1: Training and validation accuracy

![Figure 2: Training and validation loss](https://github.com/wmichalak/product_category_classifier/raw/master/data/1-loss.png)
Figure 2: Training and validation loss

On this first attempt, I already achieve 97% accuracy on the validation set. It does not appear that I am overfitting yet and 
I don't appear to have achieved a peak in the performance, o, I continued training to see how much better I could get the model with
20 more epochs.

A small, but meaningful gain in accuracy, is achieved by continuing 20 more epochs; we now have a validation accuracy of 1 and
a training accuracy of 99.05%.

![Figure 3: Training and validation accuracy](https://github.com/wmichalak/product_category_classifier/raw/master/data/2-accuracy.png)
Figure 3: Training and validation accuracy

![Figure 4: Training and validation loss](https://github.com/wmichalak/product_category_classifier/raw/master/data/2-loss.png)
Figure 4: Training and validation loss

#### 7. Assessing the performance of the model on the test set ####

Now on to the test set....

We can assess the performance of the test set using the Keras evaluate_generator

```markdown
 test_loss, test_acc = model.evaluate_generator(test_generator, steps = 50)
        print('test acc:', test_acc)
```

I receive an accuracy on the test set of 89.9%. This is ok for a first pass, but not great. 


#### 8. Improving using a convolutional base: VGG16 #### 

If I were to continue, my next step would be to use a pre-trained convolutional base, for example VGG16, which was trained
on ImageNet. I would then hold the base constant and train the last layer. The VGG16 layer convolutional layer is as follows:

```markdown
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 200, 200, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 200, 200, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 200, 200, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 100, 100, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 100, 100, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 100, 100, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 50, 50, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 50, 50, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 50, 50, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 50, 50, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 25, 25, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 25, 25, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 25, 25, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 25, 25, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 12, 12, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 12, 12, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 12, 12, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 12, 12, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 6, 6, 512)         0         
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
```

#### Final Thoughts ####

Ultimately, the model I created does not perform particularly well on the test dataset. Next, I would dive deeper into the cases that 
were incorrectly predicted - there is an underlying issue that I need to work through: perhaps the sample space in the training dataset
does not have similar enough images, the poses of the models in the Jcrew data are different than the people standing straight and facing front, or
(this is my primary guess) the the models in the test set were wearing multiple pieces of clothing. The model is likely seeing both
jeans and shirts and picking one or the other.  After looking into these issues, I would next try a convolutional base that was built on a broader data 
as mentioned in the previous section.

In the future, I would also look into how to implement a mechanism to assign to the __other__ category if the probability 
for the other categories are below a threshold. 
