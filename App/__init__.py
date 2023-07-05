from flask import Flask


from config import *
from tensorflow_addons.layers import InstanceNormalization

import os


app = Flask(__name__)

loaded_model = tf.keras.models.load_model(model_path, custom_objects={'InstanceNormalization': InstanceNormalization})

DOWNLOAD_FOLDER_ = os.path.join('App', 'static')
DOWNLOAD_FOLDER = os.path.join(DOWNLOAD_FOLDER_, 'Download')

DOWNLOAD_FOLDER_INPUT = os.path.join(DOWNLOAD_FOLDER, "input")
DOWNLOAD_FOLDER_OUTPUT = os.path.join(DOWNLOAD_FOLDER, "output")


STATIC_DOWNLOAD_FOLDER_OUTPUT = os.path.join("Download", "output")
STATIC_DOWNLOAD_FOLDER_INPUT = os.path.join("Download", "input")

from App import views

