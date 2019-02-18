'''File to check that image opens'''
import os
import matplotlib.image as mpimg


# Train data
train_files = os.walk('../data/val_data', topdown=True)

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
