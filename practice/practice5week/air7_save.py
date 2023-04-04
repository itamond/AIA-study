import pandas as pd
import numpy as np
import imgaug.augmenters as iaa

# Load train data
path='./_data/air/'
train_data = pd.read_csv(path+'train_data.csv')

# Define image augmentation pipeline
seq = iaa.Sequential([
    iaa.Flipud(p=0.5),  # flip vertically with probability 0.5
    iaa.Affine(rotate=(-10, 10), scale=(0.8, 1.2))  # rotate by -10 to 10 degrees and scale by 0.8 to 1.2
])

# Augment data
n_augmentations = 10  # number of times to augment each sample
augmented_data = []
for _, row in train_data.iterrows():
    for i in range(n_augmentations):
        img = np.array(row.values[1:]).reshape((28, 28))  # reshape to 2D image
        img_aug = seq(image=img)  # apply augmentation
        row_aug = pd.DataFrame({'id': row['id'], 'image': img_aug.flatten()})  # flatten and create new row
        augmented_data.append(row_aug)

# Combine original and augmented data
train_data_augmented = pd.concat([train_data] + augmented_data, ignore_index=True)

# Save augmented data
train_data_augmented.to_csv(path+'train_data_augmented.csv', index=False)