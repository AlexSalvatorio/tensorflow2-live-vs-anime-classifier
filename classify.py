"""
Tensorflow script from this tutorial:
https://towardsdatascience.com/cat-dog-or-elon-musk-145658489730
"""
import tensorflow as tf
import PIL

#########################
# Helpers function
#########################

import numpy as np
def load_image(img_file, target_size=(224,224)):
    X = np.zeros((1, *target_size, 3))
    X[0, ] = np.asarray(tf.keras.preprocessing.image.load_img(
        img_file, 
        target_size=target_size)
    )
    X = tf.keras.applications.mobilenet.preprocess_input(X)
    return X

import os
def ensure_folder_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
		

#Load the mobilenet model
model = tf.keras.applications.mobilenet.MobileNet()
model.summary()

#Trying to predict an image of dog (will be another image in my case)
dog_image_id = os.listdir('images/dog')[0]
dog_image = load_image(os.path.join('images/dog',dog_image_id))
print(f'shape: {dog.shape}')
print(f'type: {type(dog)}')
model.predict(dog)

#Loading the MobileNet model for transfer learning
model = tf.keras.applications.mobilenet.MobileNet(
  input_shape=(224, 224, 3), 
  include_top=False, 
  pooling='avg'
)

#Adding additionnal layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dropout, Dense, Softmax)
x = Dropout(rate=0.4)(model.output)
x = Dense(3)(x)
x = Softmax()(x)
model= Model(model.inputs, x)

#Specify teh  layer to train
for layer in model.layers[:-3]:
    layer.trainable = False

#Compile the model
from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(lr=0.001),
    loss='categorical_crossentropy'
)

#Build the data generator
from tf.keras.applications import mobilenet as _mobilenet

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=_mobilenet.preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1
)

ensure_folder_exists('training_aug')
training = datagen.flow_from_directory( 
    './images', 
    target_size=(224, 224),
    save_to_dir='./training_aug',
    subset='training'
) 

ensure_folder_exists('validation_aug')
validation = datagen.flow_from_directory( 
    './images',
    target_size=(224, 224),
    save_to_dir='./validation_aug',
    subset='validation'
)

#Train the model
from keras_tqdm import TQDMNotebookCallback

batch_size = 32

history = model.fit_generator(
    generator=training,
    steps_per_epoch=training.samples // batch_size,
    epochs=10,
    callbacks=[TQDMNotebookCallback(leave_inner=True, leave_outer=True)],
    validation_data=validation,
    validation_steps=validation.samples // batch_size
)

#Display the graph
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

#Making predictions
elon_with_disguise = load_image('elon_with_disguise.png')
elon_without_disguise = load_image('elon_no_disguise.jpg')
random_cat = random.choice(os.listdir('images/cat/'))
cat_path = os.path.join('images/cat',random_cat)
cat = load_image(cat_path)
random_dog = random.choice(os.listdir('images/dog/'))
dog_path = os.path.join('images/dog',random_dog)
dog = load_image(dog_path)

tf.keras.preprocessing.image.load_img('elon_with_disguise.png', target_size=(224,224))
tf.keras.preprocessing.image.load_img('elon_no_disguise.jpg', target_size=(224,224))
tf.keras.preprocessing.image.load_img(cat_path, target_size=(224,224))
tf.keras.preprocessing.image.load_img(dog_path, target_size=(224,224))
