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

import re
import pandas as pd

INDEX_PAGE = 'index.html'
MANOFBEAUTY_PAGE = 'ManofBeauty.html'
UPLOAD_FOLDER = 'uploaded_images/'

leaderboard = pd.DataFrame(columns=['name', 'image_url', 'score'])

class Home(RequestHandler):

    def get(self):
        try:
            self.render(INDEX_PAGE, image_src='', data={})
        except Exception as ex:
            print(ex)
            self.write("An error occurs")

class ManOfBeauty(RequestHandler):

    def get_score(self, mat):
        return np.random.randint(0, 10)

    def get(self):
        try:
            self.render(MANOFBEAUTY_PAGE, image_src='', data={})
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
            mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)   # numpy array input
            
            # save image for later use
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            image_path = os.path.join(UPLOAD_FOLDER, f'{name}.jpg')

            cv2.imwrite(image_path, mat)
            # ============ USE MODEL ================
            # step 1: calculate the score
            try:
                score = self.get_score(mat)
            except:
                score = np.random.randint(0, 10)

            # step 2: append image path and score to leaderboard
            leaderboard = leaderboard.append(pd.DataFrame({'name':[name], 'image_url':[image_path], 'score':[score]}))

            # step 3: sort leaderboard by score
            leaderboard = leaderboard.sort_values('score', ascending=False)
            print(leaderboard)
            # step 4: re-render HTML
            
            # =======================================
            
            
            self.render(MANOFBEAUTY_PAGE, image_src=image_path, data={})
        except Exception as ex:
            print('Exception', ex)
            self.render(MANOFBEAUTY_PAGE, image_src='', data={})

def make_app():
    routes = [(r'/', Home),
              (r'/manofbeauty', ManOfBeauty),
              (r'/(?:images)/(.*)', tornado.web.StaticFileHandler, {'path': './images'})]
    return Application(routes)


if __name__ == '__main__':
    app = make_app()
    print('Start serving')

    port = 7077

    print(f'Start server with  f{port}')
    app.listen(port)
    IOLoop.current().start()
