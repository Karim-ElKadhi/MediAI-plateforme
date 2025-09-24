import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import layers
from keras.layers import GlobalAveragePooling2D
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, BatchNormalization

# Reading and setup data
import cv2
import imutils
from tqdm import tqdm
import os
from tensorflow.keras.utils import to_categorical

# Loading datasets
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from extractor import process, extract_contour

SOURCE_DATASET = "path_to_your_dataset"
PROCESSED_DATASET = "path_to_your_processed_dataset"
IMG_SIZE = 224
TEST_SIZE = 0.2

from tensorflow.keras.models import load_model
model = load_model("brain_tumor_model_v2.h5")

def predict_sample(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img=extract_contour(img)
    plt.imshow(img)
    img = np.array(img, dtype="float32") / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    print(pred)
    print(class_idx)
    return "Yes (Tumor)" if class_idx == 1 else "No (Healthy)"



print(predict_sample("path_to_your_image", model))