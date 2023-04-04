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

DATA_ROOT = '../data/'
IMAGE_ROOT = '/media/sasank/LinuxStorage/Dropbox (UFL)/FistulaData/Segmentations/'