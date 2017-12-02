import csv
import cv2
import numpy as np

lines = []
with open('./data/track1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/track1/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)

    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)
print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Lambda, Dropout, Activation, BatchNormalization
from keras.optimizers import Adam

print('Keras Version: {}'.format(keras.__version__))

model = Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160, 320, 3)))

# Output: 78 x 158 x 10
model.add(Conv2D(filters=10, kernel_size=[5,5], strides=[2,2], padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))

# Output: 37 x 77 x 15
model.add(Conv2D(filters=15, kernel_size=[5,5], strides=[2,2], padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))

# Output: 18 x 38 x 20
model.add(Conv2D(filters=20, kernel_size=[3,3], strides=[2,2], padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))

# Input: 13680
model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(256))
model.add(Dense(1))

adam = Adam(lr=0.001, decay=0.01)
model.compile(loss='mean_squared_error', optimizer=adam)
model.fit(X_train, y_train, batch_size = 128, epochs = 10, 
    validation_split = 0.2, shuffle=True)

model.save('model.h5')

