# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 17:28:58 2020

@author: limz
"""

#::: Import modules and packages :::
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Import Keras dependencies
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from keras.preprocessing import image

# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL
import os

#::: Flask App Engine :::
# Define a Flask app
app = Flask(__name__)

# ::: Prepare Keras Model :::
# Model files
MODEL_ARCHITECTURE = 'Covid.json'
MODEL_WEIGHTS = 'model_covid.h5'

# Load the model from external files
json_file = open('Covid.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model= model_from_json(loaded_model_json )

# Get weights into the model
loaded_model.load_weights('Covid.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


# ::: MODEL FUNCTIONS :::
def model_predict(img_path, model):
	'''
		Args:
			-- img_path : an URL path where a given image is stored.
			-- model : a given Keras CNN model.
	'''

	IMG = image.load_img(img_path).convert('L')
	print(type(IMG))

	# Pre-processing the image
	IMG_ = IMG.resize((257, 342))
	print(type(IMG_))
	IMG_ = np.asarray(IMG_)
	print(IMG_.shape)
	IMG_ = np.true_divide(IMG_, 255)
	IMG_ = IMG_.reshape(1, 342, 257, 1)
	print(type(IMG_), IMG_.shape)

	print(model)

	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
	prediction = model.predict_classes(IMG_)

	return prediction


# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():

	# Constants:
	classes = {'TRAIN': ['COVID', 'NormalL'],
	           'VALIDATION': ['COVID', 'Normal'],
	           'TEST': ['COVID', 'Normal']}

	if request.method == 'POST':

		# Get the file from post request
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		# Make a prediction
		prediction = model_predict(file_path, model)

		predicted_class = classes['TRAIN'][prediction[0]]
		print('We think that is {}.'.format(predicted_class.lower()))

		return str(predicted_class).lower()

if __name__ == '__main__':
	app.run(debug = True)