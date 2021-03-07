import time

import tornado
from imageio import imread
from tornado.web import RequestHandler, Application
from tornado.ioloop import IOLoop
import json
import cv2
from PIL import Image
import base64
import io
import numpy as np

import sys
import os 
import glob
import re
import pandas as pd

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Input
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Model

INDEX_PAGE = 'index.html'
MANOFBEAUTY_PAGE = 'ManofBeauty.html'
UPLOAD_FOLDER = 'uploaded_images/'

leaderboard = pd.DataFrame(columns=['name', 'image_url', 'score'])

# model_paths = glob.glob('models/*.hdf5')
model_paths = ['models/5EffNet_Beauty_17--0.5673.hdf5']
models = []

for f in model_paths:
    back_bone = EfficientNetB4(include_top=False,
                weights=None,
                input_tensor=Input(shape=(320 ,320,3)),
                input_shape=(320,320,3))
    # flat = Flatten()(res_model.output)
    avg_layer = GlobalAveragePooling2D()(back_bone.output)
    final_layer = Dense(1, activation='relu')(avg_layer)
    final_model = Model(inputs=back_bone.input, outputs=final_layer)

    final_model.load_weights(f)
    models.append(final_model)

class Home(RequestHandler):

    def get(self):
        try:
            self.render(INDEX_PAGE, image_src='', data={})
        except Exception as ex:
            print(ex)
            self.write("An error occurs")

class ManOfBeauty(RequestHandler):

    def get_score(self, mat):
        p_mat = cv2.resize(mat, (320, 320))
        p_mat = p_mat[None,::]
        print(p_mat.shape)
        score = 0
        for model in models:
            score += model.predict(p_mat)[0][0]
        return score / len(models)

    def get(self):
        try:
            self.render(MANOFBEAUTY_PAGE, image_src='', data={'leaderboard':leaderboard})
        except Exception as ex:
            print(ex)
            self.write("An error occurs")

    def post(self, *args, **kwargs):
        global leaderboard
        
        if len(self.request.files) == 0:
            print(self.request.body)
            print('No image upload')
            self.render(MANOFBEAUTY_PAGE, image_src='', data={})
            return

        try:
            file_body = self.request.files['image'][0]['body']
            # name = self.request.arguments['name'][0].decode()
            name = 'fake'
                
            mat = imread(io.BytesIO(file_body))
            
            # save image for later use
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            image_path = os.path.join(UPLOAD_FOLDER, f'{name}.jpg')

            cv2.imwrite(image_path, cv2.cvtColor(mat, cv2.COLOR_RGB2BGR))
            # ============ USE MODEL ================
            # step 1: calculate the score
            try:
                score = self.get_score(mat)
            except Exception as ex:
                print(ex)
                score = np.random.normal(0.3, 1)

            # step 2: append image path and score to leaderboard
            leaderboard = leaderboard.append(pd.DataFrame({'name':[name], 'image_url':[image_path], 'score':[score]}))

            # step 3: sort leaderboard by score
            leaderboard = leaderboard.sort_values('score', ascending=False)
            print(leaderboard)

            # step 4: re-render HTML
            data = {'leaderboard':leaderboard}
            self.render(MANOFBEAUTY_PAGE, image_src=image_path, data=data)
            # =======================================
            
        except Exception as ex:
            print('Exception', ex)
            self.render(MANOFBEAUTY_PAGE, image_src='', data={})

def make_app():
    routes = [(r'/', Home),
              (r'/manofbeauty', ManOfBeauty),
              (r'/(?:images)/(.*)', tornado.web.StaticFileHandler, {'path': './images'}),
              (r'/(?:fonts)/(.*)', tornado.web.StaticFileHandler, {'path': './fonts'}),
              (r'/(?:icons)/(.*)', tornado.web.StaticFileHandler, {'path': './icons'}),
              (r'/(?:css)/(.*)', tornado.web.StaticFileHandler, {'path': './css'})]
    return Application(routes)

import asyncio

if __name__ == '__main__':
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    app = make_app()
    print('Start serving')

    port = 7777

    print(f'Start server with  f{port}')
    app.listen(port)
    tornado.ioloop.IOLoop.current().start()
