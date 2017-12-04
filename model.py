import csv
import cv2
import math
import numpy as np


def read_samples_from_csv(csv_file, samples):
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for sample in reader:
            samples.append(sample)

from sklearn.utils import shuffle
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(0, 3):
                    steering_bias = 0.0
                    if i == 1:
                        steering_bias = 0.2
                    elif i == 2:
                        steering_bias = -0.2

                    image_path = batch_sample[i]
                    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                    angle = steering_bias + float(batch_sample[3])
                    images.append(image)
                    angles.append(angle)

                    # Flip Image
                    image_flipped = np.fliplr(image)
                    angle_flipped = - angle
                    images.append(image_flipped)
                    angles.append(angle_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

#---------------------------------------
#  Generate Training, Validation and 
#  Testing data.
#--------------------------------------
track1_csv_file = './data/track1/driving_log.csv'
reversed_track1_csv_file = './data/track1_reversed/driving_log.csv'
samples = []

read_samples_from_csv(track1_csv_file, samples)
read_samples_from_csv(reversed_track1_csv_file, samples)

from sklearn.model_selection import train_test_split
train_samples, test_samples = train_test_split(samples, test_size=0.1)
train_samples, validation_samples = train_test_split(train_samples, test_size=0.2)
print('Training sample size: 6 x {}'.format(len(train_samples)))
print('Validation sample size: 6 x {}'.format(len(validation_samples)))
print('Testing sample size: 6 x {}'.format(len(test_samples)))

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
test_generator = generator(test_samples, batch_size=32)

#-----------------------------------------
#  Model Definition
#-----------------------------------------
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Lambda, Dropout, Activation, BatchNormalization, Cropping2D
from keras.optimizers import Adam

print('Keras Version: {}'.format(keras.__version__))

model = Sequential()
# Crop image
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))

model.add(Conv2D(filters=24, kernel_size=[5,5], strides=[2,2], padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))

model.add(Conv2D(filters=36, kernel_size=[5,5], strides=[2,2], padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))

model.add(Conv2D(filters=48, kernel_size=[3,3], strides=[2,2], padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))

model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(1))


#------------------------------------
#  Train model
#------------------------------------
adam = Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=adam)
hist  = model.fit_generator(
    generator = train_generator,
    steps_per_epoch = math.ceil(len(train_samples) / 32.0),
    validation_data = validation_generator,
    validation_steps = math.ceil(len(validation_samples) / 32.0),
    epochs = 20, 
)
model.save('model.h5')
print('Training History: ')
print(hist.history)


#---------------------------------------
#   Predict on test data 
#----------------------------------------
hist = model.predict_generator(
    generator = test_generator,    
    steps = math.ceil(len(test_samples) / 32.0),
)
print('Test History: ')
print(hist.history)


