import csv
import cv2
import numpy as np

def read_csv(root_path):
    with open(root_path + 'driving_log.csv', 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            yield line

def make_local_path(path, local_root):
    basename = path.split('/')[-1]
    return local_root + 'IMG/' + basename

def load_image(local_path):
    return cv2.imread(local_path)

def extract_images(csv_line, local_root):
    paths = csv_line[0:3]
    return [load_image(make_local_path(p, local_root)) for p in paths]

def extract_measurements(csv_line):
    return float(csv_line[3])

images = []
measurements = []

forward_img = './data/forward/'
reverse_img = './data/reverse/'

print("Reading forward images...\n")
for line in read_csv(forward_img):
    images.append(extract_images(line, forward_img)[0])
    measurements.append(extract_measurements(line))

print("Reading reverse images...\n")
for line in read_csv(reverse_img):
    images.append(extract_images(line, reverse_img)[0])
    measurements.append(extract_measurements(line))

print(len(images), len(measurements))
    
print("Training model...")

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

