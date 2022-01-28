#!/usr/bin/env python
# coding: utf-8

# In[63]:


# import the necessary packages
#from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2
import os
import time
from keras.models import load_model


# In[64]:


# import sys
# sys.path.append("E:\Python 3.7\Lib\site-packages")
# print(sys.path)


# In[65]:


model = load_model('weight.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[66]:


def get_xval(answer): # dạng của s (image, [x, y, w, h])
    return answer[1][0]

def is_rect(cnt):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
    #if our approximated contour has four points,
    # then we can assume we have found the paper
    if len(approx) == 4:
        return True
    return False

def show_img(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    
def show_resized_img(img, width, height):
    resized_image = cv2.resize(img, (width, height))
    cv2.imshow('img', resized_image)
    cv2.waitKey(0)


# In[67]:


def resize_img_data(img):
    img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
    img = img.reshape((28, 28, 1))
    img = np.reshape(img,[1,28,28,1])
    return img


# In[ ]:





# In[68]:


def thresh_img(image):

    # ADAPTIVE_THRESH_MEAN_C
    # ADAPTIVE_THRESH_GAUSSIAN_C

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     blurred = cv2.medianBlur(gray, 5)
#     thresh = cv2.Canny(blurred, 75, 200)

#     thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    
#     thresh = cv2.dilate(thresh,None,iterations = 1)
#     thresh = cv2.erode(thresh,None,iterations = 1)
    
    sx = cv2.Sobel(thresh, cv2.CV_32F, 1, 0)
    sy = cv2.Sobel(thresh, cv2.CV_32F, 0, 1)
    m = cv2.magnitude(sx, sy)
    thresh = cv2.normalize(m, None, 0.0, 255.0, cv2.NORM_MINMAX, cv2.CV_8U)
    

#     show_img(gray)
#     show_img(blurred)
#     show_img(thresh)
    
    return thresh


# In[69]:


def retr_codetest_id(code_test_img):
    
    #new method 27/01/2022
    
    code_id = {}
    
    legal_filled = 1
    
    code_test_img = cv2.cvtColor(code_test_img, cv2.COLOR_BGR2GRAY)
    code_test_img = cv2.threshold(code_test_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    code_test_cols = np.array_split(code_test_img, 3, 1)

    for i, cols in enumerate(code_test_cols):
#         show_img(cols)


        filled_idx = []
        choice = []
        
        code_id_bubble = np.array_split(cols, 10, 0)
        for j, b in enumerate(code_id_bubble):
            

            temp_bb = resize_img_data(b.copy())
            classes = model.predict_on_batch(temp_bb)
            classes = list(classes[0])
            
            if(classes[1] > 0.9):
                filled_idx.append(j)                
        
        if len(filled_idx) > legal_filled:
            choice = 'illegal'
        elif len(filled_idx) == 0:
            choice = "blank"
        else:
            choice = filled_idx
            
        code_id[str(i + 1)] = choice
    
    
    #old method
    
#     code_id = {}
    
#     code_test_img = cv2.cvtColor(code_test_img, cv2.COLOR_BGR2GRAY)
#     code_test_img = cv2.threshold(code_test_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
#     code_test_cols = np.array_split(code_test_img, 3, 1)

#     for i, cols in enumerate(code_test_cols):
# #         show_img(cols)

#         max_nonz = 0
#         filled_idx = -1
        
#         code_id_bubble = np.array_split(cols, 10, 0)
#         for j, b in enumerate(code_id_bubble):
# #             show_resized_img(b, 250, 250)
# #             show_img(b)
#             curr_nonz = cv2.countNonZero(b)
#             if curr_nonz > max_nonz:
#                 filled_idx = j
#                 max_nonz = curr_nonz    
#         code_id[str(i + 1)] = str(filled_idx)
    return code_id


# In[70]:


def retr_student_id(student_id_img):
    
    student_id = {}
    
    legal_filled = 1
    
    
    student_id_img = cv2.cvtColor(student_id_img, cv2.COLOR_BGR2GRAY)
    student_id_img = cv2.threshold(student_id_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    student_id_cols = np.array_split(student_id_img, 6, 1)
    
    for i, cols in enumerate(student_id_cols):

        
        student_id_bubble = np.array_split(cols, 10, 0)

        filled_idx = []
        choice = []

        for j, b in enumerate(student_id_bubble):
            
#             print('student id')
#             print(i + 1 , j + 1)
#             show_img(b)
            
            temp_bb = resize_img_data(b.copy())
            classes = model.predict_on_batch(temp_bb)
            classes = list(classes[0])
            
            if(classes[1] > 0.9):
                filled_idx.append(j)
                
#                 print('student id')
#                 print(i + 1 , j + 1)
#                 show_img(b)

                
        if len(filled_idx) > legal_filled:
            choice = 'illegal'
        elif len(filled_idx) == 0:
            choice = "blank"
        else:
            choice = filled_idx
            
        student_id[str(i + 1)] = choice
    
    
    
    #old method
    
#     student_id = {}
    
#     student_id_img = cv2.cvtColor(student_id_img, cv2.COLOR_BGR2GRAY)
#     student_id_img = cv2.threshold(student_id_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
#     student_id_cols = np.array_split(student_id_img, 6, 1)
    
#     for i, cols in enumerate(student_id_cols):
# #         show_img(cols)

#         max_nonz = 0
#         filled_idx = -1
        
#         student_id_bubble = np.array_split(cols, 10, 0)
#         for j, b in enumerate(student_id_bubble):
# #             show_resized_img(b, 250, 250)
# #             show_img(b)
#             curr_nonz = cv2.countNonZero(b)
#             if curr_nonz > max_nonz:
#                 filled_idx = j
#                 max_nonz = curr_nonz    
#         student_id[str(i + 1)] = str(filled_idx)
    return student_id


# In[71]:


def get_choice(argument):
    switcher = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
    }
    return switcher.get(argument, "blank")


# In[72]:


def get_answer(answer_list):
    
    key_gen = {}
    
    legal_filled = 1
    
    for i, ans in enumerate(answer_list):
        ans = cv2.cvtColor(ans, cv2.COLOR_BGR2GRAY)
        ans = cv2.threshold(ans, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        bubble_row = np.array_split(ans, 5, 1)
        bubble_row.pop(0)
        
        
        filled_idx = []
        choice = []
        
        for j, b in enumerate(bubble_row):
            temp_bb = resize_img_data(b.copy())
            classes = model.predict_on_batch(temp_bb)
            classes = list(classes[0])
            
            if(classes[1] > 0.9):
                filled_idx.append(j)
        
        if len(filled_idx) > legal_filled:
            choice = 'illegal'
        elif len(filled_idx) > 0:
            for idx in filled_idx:
                choice.append(get_choice(idx))
        else:
            choice = "blank"
        
        key_gen[str(i + 1)] = choice
    
    
    #old method
    
#     key_gen = {}
    
#     for i, ans in enumerate(answer_list):
#         ans = cv2.cvtColor(ans, cv2.COLOR_BGR2GRAY)
#         ans = cv2.threshold(ans, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
#         arr_choice = []
        
# #         show_img(ans)
#         bubble_row = np.array_split(ans, 5, 1)
#         bubble_row.pop(0)
        
#         max_nonz = 0
#         choice = ''
#         filled_idx = -1
#         for j, b in enumerate(bubble_row):
#             curr_nonz = cv2.countNonZero(b)
# #             show_img(b)
#             if curr_nonz > max_nonz:
#                 filled_idx = j
#                 max_nonz = curr_nonz
                
#         choice = get_choice(filled_idx)
        
#         arr_choice.append(choice)
#         key_gen[str(i + 1)] = arr_choice
    return key_gen


# In[73]:


def retr_choice(sorted_ans_blocks):
    #collect answer
    box_distance = 10

    answer_list = []
    key_gen = []
    
    for ans_block in sorted_ans_blocks:
        img = np.array(ans_block[0]) #take each answer block image

        #take each answer box
        box_img_rows = np.array_split(img, 6, 0)
        
        for tmp_img in box_img_rows: 
            img = tmp_img[box_distance:tmp_img.shape[0]-box_distance, :]
            
#             show_img(img)
            answer_rows = np.array_split(img, 5, 0)
            for j in answer_rows:
                answer_list.append(j)
#                 show_img(j)
                
    key_gen = get_answer(answer_list)
    return key_gen


# In[ ]:





# In[74]:


def retrInfo(org_img):
    # find contours in the edge map, then initialize
    # the contour that corresponds to the document
    result = {}
    image = org_img.copy()
    
    thresh = thresh_img(image)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)

    answer_block = []

    # ensure that at least one contour was found
    if len(cnts) > 0:
        # sort the contours according to their size in
        # descending order

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        
#         cv2.drawContours(img, cnts, -1, (0, 255, 0), 3)
#         show_img(img)
        
    
        answer_block_coor = cnts[0:4]

        student_id_coor = cnts[6]
        code_test_coor = cnts[7]

        # loop over the sorted contours
        for c in answer_block_coor:
            # approximate the contour

            x,y,w,h = cv2.boundingRect(c)
            answer_block.append((image[y:y + h, x:x + w], [x, y, w, h]))
#             show_img(image[y:y + h, x:x + w])

            #collect answer block
        sorted_ans_blocks = sorted(answer_block, key=get_xval)
        
        key_gen = retr_choice(sorted_ans_blocks)
        
        x,y,w,h = cv2.boundingRect(student_id_coor)
        student_id_img = image[y:y + h, x:x + w]
        
#         show_img(student_id_img)
        student_id_block = retr_student_id(student_id_img)
            
        x,y,w,h = cv2.boundingRect(code_test_coor)
        code_test_img = image[y:y + h, x:x + w]
#         show_img(code_test_img)
        
        code_id = retr_codetest_id(code_test_img)
    
        student_str = ''
        for value in student_id_block.values():
            if value == 'blank' or value == 'illegal':
                student_str += '-'
            else: 
                student_str += str(value[0])
                
        code_str = ''
        for value in code_id.values():
            if value == 'blank' or value == 'illegal':
                code_str += '-'
            else: 
                code_str += str(value[0])
    
    
    
    
    result['answer'] = key_gen
    result['student_id'] = student_str
    result['code_id'] = code_str
#     result.append(key_gen)
#     result.append(student_id_block)
#     result.append(code_id)
    return result


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[75]:


# start_time = time.time()


# In[76]:


# for i in range(200):
#     filename = 'answer_gen/ans_' + str(i + 1) + '.png'
#     image = cv2.imread(filename)
    
#     print('---------------Scanning img ' + str(i + 1) + '--------------------------')
#     retrInfo(image)


# In[77]:


# print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:





# In[ ]:





# In[ ]:





# # API 

# In[78]:


from flask import Flask, request, redirect, jsonify
import urllib
import numpy as np
import json

from functools import wraps
from flask_restful import Resource, Api, reqparse
import werkzeug
from werkzeug.utils import secure_filename
import pandas as pd
import ast

app = Flask(__name__)
api = Api(app)


# In[79]:


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image


# In[80]:


# app.url_map


# In[81]:


class UploadImage(Resource):
    def post(self):
        result = {}
        url = request.get_data()
        url_js = json.loads(url)
#         print('data sending : ' + str(url_js))

        image = url_to_image(url_js['url'])
        result = retrInfo(image)
        
        url_js['result'] = result
        return url_js


# In[82]:


api.add_resource(UploadImage, '/api')


# In[83]:

if __name__ == '__main__':
    app.run()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




