
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='C:/Users/91700/dl deploy/signal_model2.h5'

# Load your trained model
model = load_model(MODEL_PATH)

classes = { 0:'No passing',
            1:'No passing veh over 3.5 tons',
            2:'Right-of-way at intersection',
            3:'Priority road',
            4:'Yield',
            5:'Stop',
            6:'Vehicle > 3.5 tons prohibited',
            7:'No entry',
            8:'General caution',
            9:'Dangerous curve left',
            10:'Dangerous curve right',
            11:'Bumpy road',
            12:'Slippery road',
            13:'Road narrows on the right',
            14:'Road work',
            15:'Traffic signals',
            16:'Pedestrians',
            17:'End speed + passing limits',
            18:'Go straight or left',
            19:'End of no passing',
            20:'End no passing vehicle > 3.5 tons' }


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    images=np.vstack([x])
    val=model.predict(images)
    value=val[0]
    if 1 not in val[0]:
        print("No data present")
    else:
        print("image is of :   ", classes[np.where(value==1)[0][0]])
        string = "image is of  " + str(classes[np.where(value==1)[0][0]])
    
    return string


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)