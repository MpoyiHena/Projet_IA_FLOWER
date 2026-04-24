import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import cv2
import keras
from keras import layers
#from tensorflow import data as tf_data
import os

#https://keras.io/api/data_loading/image/

def traitement_image():
    # Code pour le traitement d'image
    train=keras.utils.image_dataset_from_directory(r"C:\Dossier E\Henallux\M1\Systeme inteligent\Projet\flowers\Dataset_flowers\Training Data",labels="inferred",batch_size=15746,image_size=(256,256),format="tf")
    test=keras.utils.image_dataset_from_directory(r"C:\Dossier E\Henallux\M1\Systeme inteligent\Projet\flowers\Dataset_flowers\Testing Data",labels="inferred",batch_size=2460,image_size=(256,256),format="tf")

    class_names = train.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in train.take(1):
        for i in range(min(9, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(class_names[int(labels[i])])
            plt.axis("off")
    plt.show()


traitement_image()




#Data augmentation pour éviter le surapprentissage (Ajouter Flipping, Rotation, décalage, etc...)

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images



#// Peut etre mieux ceci pour économiser de la GPU :

inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
...  # Rest of the model




#Normalisation des images pour que les valeurs soient entre 0 et 1

print('Before normalization : Min={}, max={}'.format(x_train.min(),x_train.max()))

xmax=x_train.max()
x_train = x_train / xmax
x_test  = x_test  / xmax

print('After normalization  : Min={}, max={}'.format(x_train.min(),x_train.max()))










#Création du modèle CNN

#hidden1     = 100
#hidden2     = 100

model = keras.models.Sequential()

#model.add( keras.layers.Input((28,28,1))) #input layer

model.add( keras.layers.Conv2D(16, (5,5),  activation='relu', padding="same", input_shape = (28,28,1)) )#8 2dconvolutiv planes, kernel size 3*3
model.add( keras.layers.MaxPooling2D((2,2)))#reduce the image size on 4
# model.add( keras.layers.Dropout(0.2))#deactivate randomely some neuron outputs (regularization and avoid overfitting)

model.add( keras.layers.Conv2D(32, (5,5), activation='relu', padding="same") )
model.add( keras.layers.MaxPooling2D((2,2)))
# model.add( keras.layers.Dropout(0.2))

model.add( keras.layers.Conv2D(64, (5,5), activation='relu', padding="same") )
model.add( keras.layers.MaxPooling2D((2,2)))
# model.add( keras.layers.Dropout(0.2))

model.add( keras.layers.Flatten()) 
model.add( keras.layers.Dense(128, activation='relu'))
model.add( keras.layers.Dropout(0.3))

model.add( keras.layers.Dense(64, activation='relu'))
model.add( keras.layers.Dropout(0.5))

model.add( keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',#function for get down the gradient
              loss='sparse_categorical_crossentropy',#loss function for classification
              metrics=['accuracy']) #the metric 










#Training du modèle CNN

batch_size  = 600
epochs      =  10


history = model.fit(  x_train, y_train,
                      batch_size      = batch_size,
                      epochs          = epochs,
                      verbose         = 1,
                      validation_data = (x_test, y_test))











#Evaluation du modèle CNN

score = model.evaluate(x_test, y_test, verbose=0)

print(f'Test loss     : {score[0]:4.4f}')
print(f'Test accuracy : {score[1]:4.4f}')



