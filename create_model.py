import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D, Flatten, Dense, Dropout
import pickle 
import matplotlib.pyplot as plt
from keras.models import model_from_json

Epochs = 5


training_datagen = ImageDataGenerator( rescale = 1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = training_datagen.flow_from_directory('train' , target_size=(64,64) , batch_size=32, class_mode='binary')

test_dataset = test_datagen.flow_from_directory('test', target_size=(64,64) , batch_size=32, class_mode='binary')


model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3) , input_shape = (64,64,3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=(3,3) , activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1 , activation='sigmoid'))
model.compile(optimizer='adam' , loss='binary_crossentropy' , metrics = ['accuracy'])
model.fit_generator(training_dataset, steps_per_epoch=1000, epochs= Epochs, validation_data=test_dataset, validation_steps=1000)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")



