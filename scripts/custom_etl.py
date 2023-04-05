"""
Sasank Desaraju
4/4/23

This is to create a train/test/val split of CSVs for our Fistula Segmentation dataset.
"""

import sys
import os
import time
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import numpy as np
import glob
import csv


"""
Each pair of image and label are inside of their own folder with a numerical identifier.
Within each folder, the image and label are named [number]_image.nii.gz and [number]_label.nii.gz.
For example, the image and label for the pair with the identifier 1 are located at:
    IMAGE_ROOT/1/1_image.nii.gz
    IMAGE_ROOT/1/1_label.nii.gz

We want to create, first, a master CSV that just contains a list of the iamges and their labels in two columns.

Then, we want to split this into train/test/val sets and create CSVs of each of the splits.
The CSVs should be in the format:
    image_name, label_name
where both the image_name and label are relative to the IMAGE_ROOT.
"""

DATA_ROOT = '/home/sasank/Documents/GitRepos/Fistula-Segmentation/data/'
IMAGE_ROOT = '/media/sasank/LinuxStorage/Dropbox (UFL)/FistulaData/Segmentations/'

def create_master_csv(master_csv_name):
    """
    Create a master CSV with the image and label names.
    """
    # Get the list of folders
    folders = glob.glob(IMAGE_ROOT + '*')
    # Create the master CSV
    with open(DATA_ROOT + master_csv_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'label', 'patient_id'])
        for folder in folders:
            # Get the image and label names
            image = os.path.join(folder.split('/')[-1], folder.split('/')[-1] + '_image.nii.gz')
            label = os.path.join(folder.split('/')[-1], folder.split('/')[-1] + '_label.nii.gz')
            patient_id = folder.split('/')[-1]
            # Write the image and label names to the CSV
            writer.writerow([image, label, patient_id])

def create_train_test_val_csvs(data_name, master_csv_name):
    """
    Create the train/test/val CSVs.
    """
    # Read in the master CSV
    df = pd.read_csv(DATA_ROOT + master_csv_name)
    # Split the data into train/test/val
    train, test = tts(df, test_size=0.2, random_state=42)
    train, val = tts(train, test_size=0.2, random_state=42)
    # Make the subdirectory for the data split using the data_name if it doesn't exist
    if not os.path.exists(DATA_ROOT + data_name):
        os.mkdir(DATA_ROOT + data_name)
    # Write the CSVs
    train.to_csv(os.path.join(DATA_ROOT, data_name, 'train_' + data_name + '.csv'), index=False)
    test.to_csv(os.path.join(DATA_ROOT, data_name, 'test_' + data_name + '.csv'), index=False)
    val.to_csv(os.path.join(DATA_ROOT, data_name, 'val_' + data_name + '.csv'), index=False)


if __name__ == '__main__':
    # Create the master CSV
    create_master_csv(master_csv_name='full_data.csv')
    # Create the train/test/val CSVs
    create_train_test_val_csvs(data_name='BaseSplit', master_csv_name='full_data.csv')