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

    def get(self):
        self.render("public/index.html", image_src='', data={})

    def post(self, *args, **kwargs):
        if len(self.request.files) == 0:
            self.render('public/index.html', image_src='', data={})
            return
        try:
            file_body = self.request.files['image'][0]['body']
                
            mat = imread(io.BytesIO(file_body))
            mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)   # numpy array input

            # ============ USE MODEL ================


            # =======================================
            
            image_path = 'public/input.jpg'
            cv2.imwrite(image_path, mat)

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
