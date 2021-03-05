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

class T4EBeauty(RequestHandler):

    def __init__(self):
        # call super contructor
        self.RequestHandler.__init__()

        # define learderboad
        self.learderboad = pd.DataFrame(columns=['name', 'image_url', 'score'])

        # def init model

    def get_score(self, mat):
        return np.random.randint(0, 10)

    def get(self):
        self.render("public/index.html", image_src='', data={})

    def post(self, *args, **kwargs):
        if len(self.request.files) == 0:
            self.render('public/index.html', image_src='', data={})
            return
        try:
            file_body = self.request.files['image'][0]['body']
            name = self.request.json['data']
                
            mat = imread(io.BytesIO(file_body))
            mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)   # numpy array input
            
            # save image for later use
            image_path = f'uploaded_images/{name}.jpg'

            cv2.imwrite(image_path, mat)
            # ============ USE MODEL ================
            # step 1: calculate the score
            score = self.get_score(mat)

            # step 2: append image path and score to leaderboad

            # step 3: sort leaderboad by score

            # step 4: re-render HTML
            
            # =======================================
            
            

            self.render('public/index.html', image_src=image_path, data={})
        except Exception as ex:
            print('Exception', ex)
            self.render('public/index.html', image_src='', data={})
      
def make_app():
    routes = [(r'/', T4EBeauty),
              (r'/(?:public)/(.*)', tornado.web.StaticFileHandler, {'path': './public'})]
    return Application(routes)


if __name__ == '__main__':
    app = make_app()
    print('Start serving')

    port = 7077

    print(f'Start server with  f{port}')
    app.listen(port)
    IOLoop.current().start()
