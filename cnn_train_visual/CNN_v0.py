# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 13:26:39 2019

@author: shihao
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

regressor = Sequential()

regressor.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

regressor.add(MaxPooling2D(pool_size = (2, 2)))

regressor.add(Flatten())

regressor.add(Dense(128, activation = 'relu'))
regressor.add(Dense(3))

regressor.compile('adam', loss = 'mean_squared_error')

from keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(validation_split=0.2)

train_generator = data_generator.flow_from_directory(TRAINING_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True, seed=13,
                                                     class_mode='categorical', batch_size=BATCH_SIZE, subset="training")

validation_generator = data_generator.flow_from_directory(TRAINING_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True, seed=13,
                                                     class_mode='categorical', batch_size=BATCH_SIZE, subset="validation")

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])