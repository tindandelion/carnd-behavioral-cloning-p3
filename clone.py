import csv
import cv2
import numpy as np

def read_csv():
    with open('data/driving_log.csv', 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            yield line

def make_local_path(path):
    basename = path.split('/')[-1]
    return './data/IMG/' + basename

def load_image(local_path):
    return cv2.imread(local_path)

def extract_images(csv_line):
    paths = csv_line[0:3]
    return [load_image(make_local_path(p)) for p in paths]

def extract_measurements(csv_line):
    return float(csv_line[3])

images = []
measurements = []

for line in read_csv():
    images.append(extract_images(line)[0])
    measurements.append(extract_measurements(line))

print(len(images), len(measurements))
    
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=15)

model.save('model.h5')

