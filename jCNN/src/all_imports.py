# import the necessary packages
# Utilities
import os
import sys
import time
import pickle
import datetime
import math, random

from itertools import cycle

#Data Analysis and Visualization
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interp
from sconf import Config
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from sklearn.calibration import calibration_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Model Development
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D

#Model Evaluation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report,accuracy_score,f1_score

#Other external packages
#!pip install git+https://github.com/qubvel/segmentation_models for eff_net
