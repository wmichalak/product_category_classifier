
### Introduction ###
The purpose of this project is to classify the category of clothing shown in an image as:
Dresses, Tops, Jeans, Skirts, Rompers, Shoes, Bags, Jewelry, Swimwear, Intimates, Others.

For this project I will used a multilabel CNN classifier based in the Keras API.

The project consists of 8 parts:
1. Acquiring the validation data and assigning labels based on the description.  
2. Acquire a train / test dataset to learn coefficients of the CNN.
3. Define image prepreprocessing
4. Define the CNN structure
5. Train the CNN
6. Assess the error based on test, train
7. Perform hyperparameter tuning
8. Perform final validation error assessment

#### 1: Acquiring the validation data and assigning labels based on the description.  ####
I choose to use the given dataset as a validation dataset for two reasons: 1) the final test of the model is based on well
the model performs on the provided set and two, the set has imbalanced classes and therefore a model would have a difficult
time training to identify the minority classes. Thus, I used fashion items pre-defined in product_data.json as a validation set only.
The file was read and stored as a dictionary using the `json`' package. Each photo was downloaded using the `urllib` package. 

```
urllib.request.urlretrieve(item['images_url'], photo_filename)
```

To assign labels to each photo, I created a list of synonyms from each category item and saved this in the categories.txt file.
I considered using the `thesaurus` package to automatically generate a list of synonyms, but found too much overlap across items;
not to mention that removing the non-fashion synonyms would be tricky, e.g. _top_.

The set of synonyms were:

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
often top was used with bikini, for example. Thus, I set a rules for 
- 'bikini' to supercede top.

To assign the labels, I broke down each `description` by tokenizing and lemmatizing and searching the category keywords in the 
description. I used the `spacy` natural language processor `nlp` for this task.

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

After determining the label, I saved file as shown above (with urllib) in a folder designated by the label.
        
A couple of odds and ends to point out:
* In general, there are weaknesses in my approach, e.g., when more than one keyword is present, the algorithm selects the first
one it checks. This is not something I am choosing to spend time on. Rather I am manually fixing the mis-labelled items.
* Since _pants_ are not a unique category (rather, jeans is used), I assign pants to the other category as __jeans__ do not necessarily
cover pants.
* some descriptions were just too vague and manual labelling was required. For example, the description:
_Shop On Top at Urban Outfitters. We have all the latest styles and fashion trends you're looking for right here._
* There were also images in this set that were not fashion items, they images were assigned to 
the other category.
* Some images could not be downloaded and thus were left out of the validation set and are designated by the nan category below.
* A summary report of the filename, assigned label, whether the download was complete, the description and url were saved in a file called
val_database.csv (which is in the repository).

In summary, out of 1000 records, there were X items with

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

#### 2: Acquiring the validation data and assigning labels based on the description.  ####
To train the model, I needed labeled training and test data. I decided that the product images from the validation set should not
be used to teaching the model. I decided to download labeled training data from Google images by searching
for the categories and downloading those images. I used the 
[google images download project](https://github.com/hardikvasa/google-images-download). In order to acquire more than 100 images for each category,
I had to download the Chrome webdriver that is used with selenium and modify line 853 in googe_images_download with the path
to the chromedriver executable. 

All of the data was downloaded to folders with corresponding labels.

#### 3. Define image prepreprocessing: ImageGenerator.flowfromdirectory() with Keras ###