import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import cv2
import keras
from keras import layers
from tensorflow import data as tf_data
import os

#https://keras.io/api/data_loading/image/

def traitement_image():
    # Code pour le traitement d'image
    train=keras.utils.image_dataset_from_directory(r"Projet\flowers\Dataset_flowers\Training Data",labels="inferred",batch_size=100,image_size=(256,256),format="tf")
    test=keras.utils.image_dataset_from_directory(r"Projet\flowers\Dataset_flowers\Testing Data",labels="inferred",batch_size=100,image_size=(256,256),format="tf")

    class_names = train.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in train.take(1):
        for i in range(min(9, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(class_names[int(labels[i])])
            plt.axis("off")
    # plt.show()



traitement_image()
    # print("x_train : ",x_train)
    # print("y_train : ",y_train)
    # print("x_test  : ",x_test)
    # print("y_test  : ",y_test)
