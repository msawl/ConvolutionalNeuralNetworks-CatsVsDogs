import keras
import numpy as np
import parser 
import glob
import numpy as np
import os.path as path
import scipy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential , load_model
from keras.layers import Conv2D , MaxPooling2D, Flatten, Dense
import tensorflow as tf
import pickle

from keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

test_image = image.load_img('4.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image , axis=0)
result = loaded_model.predict(test_image)

print(result)

if result >= 0.5:
    print('cat')
else:
    print('dog')
	