import csv
import cv2
import random as rnd
import numpy as np
import matplotlib.pyplot as plt


from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Lambda, Cropping2D, Dropout
from keras.optimizers import Adagrad

def read_csv(root_path):
    with open(root_path + 'driving_log.csv', 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            yield line

def make_local_path(path, local_root):
    basename = path.split('/')[-1]
    return local_root + 'IMG/' + basename

def read_data(data_path):
    samples = []
    for line in read_csv(data_path):
        images = [make_local_path(p, data_path) for p in line[0:3]]
        steering_angle = float(line[3])
        samples.append((images, steering_angle))
                
    return samples

def baseline_mse(samples):
    angles = np.array([s[1] for s in samples])
    return np.mean(np.square(angles - np.mean(angles)))

def augment_with_side_cameras(samples, angle_correction=0):
    new_samples = []
    for images, angle in samples:
        center, left, right = images
        new_samples.append((center, angle))
        if angle_correction > 0:
            new_samples.append((left, angle + angle_correction))
            new_samples.append((right, angle - angle_correction))
    return new_samples

def read_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    batch_size = batch_size
    while True: 
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for img_path, angle in batch_samples:
                img = read_image(img_path)
                images.append(img)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
def balance_samples(samples):
    result = []
    zero_threshold = 0.1

    for s in samples: 
        angle = s[1]
        if abs(angle) < zero_threshold:
            chance = rnd.uniform(0, 1)
            if (chance > 0.7):
                result.append(s)
        else:
            result.append(s)
    return result
            

data_path = './driving-training/'
angle_correction = 0.2

samples = balance_samples(read_data(data_path))
print("Dataset size: %d" % len(samples))

train_samples, valid_samples = train_test_split(samples, test_size=0.2)

train_samples = augment_with_side_cameras(samples, angle_correction)
valid_samples = augment_with_side_cameras(valid_samples, 0)

print("Training set size: %d" % len(train_samples))
print("Validation set size: %d" % len(valid_samples))

image_shape = (160, 320, 3)
cropping = ((60, 0), (0, 0))
dropout_rate = 0
batch_size = 128

normalize = lambda x: (x - 127.0) / 127.0

model = Sequential()
model.add(Lambda(normalize, input_shape=image_shape))
model.add(Cropping2D(cropping=cropping))
model.add(Convolution2D(24, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Convolution2D(36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Convolution2D(48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Convolution2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(Convolution2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))

model.add(Flatten())
model.add(Dropout(dropout_rate))
model.add(Dense(100, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(50, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

optimizer = Adagrad(lr=0.001)
model.compile(loss='mse', optimizer=optimizer)

history = model.fit_generator(generator(train_samples, batch_size), len(train_samples) // batch_size, 
                             validation_data=generator(valid_samples, batch_size), 
                             validation_steps=(len(valid_samples) // batch_size), 
                             epochs = 10, 
                             verbose = 1)

model.save('model.h5')
