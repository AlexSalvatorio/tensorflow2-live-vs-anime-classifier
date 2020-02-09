import tensorflow as tf
import PIL
import numpy as np
import os

from tensorflow.python.keras.callbacks import TensorBoard
from datetime import datetime


#Using Tensorboard in Jupyter
#https://www.dlology.com/blog/how-to-run-tensorboard-in-jupyter-notebook/
logs_base_dir = "./logs"
os.makedirs(logs_base_dir, exist_ok=True)
#tensorboard --logdir {logs_base_dir}

#Dir where is the dataset here
DATA_SET_DIR = '../../../Trainning/training-anime-live/dataset/'

#########################
# Helpers function
#########################

def load_image(img_file, target_size=(224,224)):
    X = np.zeros((1, *target_size, 3))
    X[0, ] = np.asarray(tf.keras.preprocessing.image.load_img(
        img_file, 
        target_size=target_size)
    )
    X = tf.keras.applications.mobilenet.preprocess_input(X)
    return X

def ensure_folder_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
		
        
#Load the mobilenet model
model = tf.keras.applications.mobilenet.MobileNet()
model.summary()

#########################
# Test the prediction
#########################

"""
#Trying to predict an image of anime (will be another image in my case)
anime_image_id = os.listdir(DATA_SET_DIR+'anime')[0]
anime_image = load_image(os.path.join(DATA_SET_DIR+'anime',anime_image_id))
print(f'shape: {anime_image.shape}')
#print(f'type: {type(anime_image)}')
#result = model.predict(anime_image)
#print(result) #--> what shape was this???
"""

#Loading the model for transfert learning
model = tf.keras.applications.mobilenet.MobileNet(
  input_shape=(224, 224, 3), 
  include_top=False,#Important, because we can add our own top layer (shape)
  pooling='avg' #Why AVG?
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dropout, Dense, Softmax)

x = Dropout(rate=0.4)(model.output) # avoid overfitting
x = Dense(2)(x) # 3 to 2, does this influence the numbe rof category? YES!
x = Softmax()(x) #Output
model= Model(model.inputs, x)

#lAyer we need to train
for layer in model.layers[:-3]:
    layer.trainable = False
    
from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(lr=0.001), #Best optimizer
    loss='categorical_crossentropy' #for categories
)

from tensorflow.keras.applications import mobilenet as _mobilenet


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=_mobilenet.preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1
)

ensure_folder_exists('training_aug')
training = datagen.flow_from_directory( 
    DATA_SET_DIR, 
    target_size=(224, 224),
    save_to_dir='./training_aug',
    subset='training'
) 

ensure_folder_exists('validation_aug')
validation = datagen.flow_from_directory( 
    DATA_SET_DIR,
    target_size=(224, 224),
    save_to_dir='./validation_aug',
    subset='validation'
)

#from keras_tqdm import TQDMNotebookCallback
batch_size = 32

#Set the Tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(logs_base_dir, histogram_freq=1)
print('Launch the model trainning!')
history = model.fit_generator(
    generator=training,
    steps_per_epoch=training.samples // batch_size,
    epochs=10,
    callbacks=[tensorboard_callback],
    validation_data=validation,
    validation_steps=validation.samples // batch_size
)