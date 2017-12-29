import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


# FUNCTIONAL DEFINITIONS
def generator(samples, batch_size=32, use_side_images = False, steer_correction = 0.2):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            

            for batch_sample in batch_samples:
                name = 'Data/IMG/'+batch_sample[0].split('/')[-1]
                image_center = cv2.imread(name)
                steering_center = float(batch_sample[3])
                # append to list:
                images.append(image_center)
                angles.append(steering_center)
                # if using with side images:
                if use_side_images:
                    # create adjusted steering measurements for the side camera images
                    steering_left = steering_center + steer_correction
                    steering_right = steering_center - steer_correction
                    img_left = process_image(np.asarray(Image.open(path + row[1])))
                    img_right = process_image(np.asarray(Image.open(path + row[2])))

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            

# Data Loading
samples = []
with open('Data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Partition Data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(ch, row, col),output_shape=(ch, row, col)))
model.add(... finish defining the rest of your model architecture here ...)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= /
            len(train_samples), validation_data=validation_generator, /
            nb_val_samples=len(validation_samples), nb_epoch=3)