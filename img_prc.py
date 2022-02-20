#!/usr/bin/env python
# coding: utf-8

# In[25]:


import cv2


# In[ ]:





# # API 

# In[26]:


from flask import Flask, request, redirect, jsonify
import urllib
import numpy as np
import json

from functools import wraps
from flask_restful import Resource, Api, reqparse
import pandas as pd

from multiprocessing import Pool
import threading

import imgprc.fst as fst

from rq import Queue
from worker import conn

q = Queue(connection=conn)

app = Flask(__name__)
api = Api(app)


# In[27]:


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image

def func_thread(image, out):
    out.append(worker.retrInfo(image))


# In[28]:


# app.url_map


# In[29]:


class UploadImage(Resource):
    def post(self):
        child_result = {}
        result  = []
        
        url = request.get_data()
        url_js = json.loads(url)

        arr_url = (url_js['url'])
        
        for url in arr_url:
            image = url_to_image(url)
            child_result = fst.retrInfo(image)
            child_result = q.enqueue(fst.retrInfo, image)        
        
            result.append(child_result)
        url_js['result'] = result



#         arr_url = (url_js['url'])
#         arr_img = []
#         for url in arr_url:
#             arr_img.append(url_to_image(url))
                    
    
#         multiprocessing
#         num_processors = 4
#         p = Pool(processes = num_processors)
#         results = [p.apply_async(worker.retrInfo, args=(img,)) for img in arr_img]
#         output = [p.get() for p in results]
#         url_js['result'] = output
        
    
        #multithread
#         thread_list = []
#         results = []
        
#         for img in arr_img:
#             thread = threading.Thread(target=func_thread, args=(img, results))
#             thread_list.append(thread)
            
            
#         for thread in thread_list:
#             thread.start()
#         for thread in thread_list:
#             thread.join()
            
#         url_js['result'] = results
        
        return url_js


# In[30]:


api.add_resource(UploadImage, '/api')


# In[31]:

if __name__ == '__main__':
    app.run()


# In[ ]:





# In[32]:


# url = []

# temp_url = "https://res.cloudinary.com/dunbjnt9i/image/upload/v1643344505/test-img/omr_org_cp3_rjrljk.png"
# for i in range(100):
#     url.append(str(temp_url))
# url


# In[ ]:





# In[ ]:




