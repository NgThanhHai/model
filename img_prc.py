#!/usr/bin/env python
# coding: utf-8

# In[158]:


# import the necessary packages
import numpy as np
import imutils
import cv2
import time


# In[159]:


# import sys
# sys.path.append("E:\Python 3.7\Lib\site-packages")
# print(sys.path)


# In[ ]:





# In[162]:


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


# In[ ]:





# In[ ]:





# In[163]:


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


# In[164]:


def retr_codetest_id(code_test_img):
    code_id = []
    
    code_test_img = cv2.cvtColor(code_test_img, cv2.COLOR_BGR2GRAY)
    code_test_img = cv2.threshold(code_test_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    code_test_cols = np.array_split(code_test_img, 3, 1)

    for i, cols in enumerate(code_test_cols):
#         show_img(cols)

        max_nonz = 0
        filled_idx = -1
        
        code_id_bubble = np.array_split(cols, 10, 0)
        for j, b in enumerate(code_id_bubble):
#             show_resized_img(b, 250, 250)
#             show_img(b)
            curr_nonz = cv2.countNonZero(b)
            if curr_nonz > max_nonz:
                filled_idx = j
                max_nonz = curr_nonz    
        code_id.append((i + 1, filled_idx))
    return code_id


# In[165]:


def retr_student_id(student_id_img):
    student_id = []
    
    student_id_img = cv2.cvtColor(student_id_img, cv2.COLOR_BGR2GRAY)
    student_id_img = cv2.threshold(student_id_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    student_id_cols = np.array_split(student_id_img, 6, 1)
    
    for i, cols in enumerate(student_id_cols):
#         show_img(cols)

        max_nonz = 0
        filled_idx = -1
        
        student_id_bubble = np.array_split(cols, 10, 0)
        for j, b in enumerate(student_id_bubble):
#             show_resized_img(b, 250, 250)
#             show_img(b)
            curr_nonz = cv2.countNonZero(b)
            if curr_nonz > max_nonz:
                filled_idx = j
                max_nonz = curr_nonz    
        student_id.append((i + 1, filled_idx))
    return student_id


# In[166]:


def get_choice(argument):
    switcher = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
    }
    return switcher.get(argument, "blank")


# In[167]:


def get_answer(answer_list):
    key_gen = []
    
    for i, ans in enumerate(answer_list):
        ans = cv2.cvtColor(ans, cv2.COLOR_BGR2GRAY)
        ans = cv2.threshold(ans, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
#         show_img(ans)
        bubble_row = np.array_split(ans, 5, 1)
        bubble_row.pop(0)
        
        max_nonz = 0
        choice = ''
        filled_idx = -1
        for j, b in enumerate(bubble_row):
            curr_nonz = cv2.countNonZero(b)
#             show_img(b)
            if curr_nonz > max_nonz:
                filled_idx = j
                max_nonz = curr_nonz
                
        choice = get_choice(filled_idx)
        key_gen.append((i + 1, choice))
    return key_gen


# In[168]:


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





# In[169]:


def retrInfo(org_img):
    # find contours in the edge map, then initialize
    # the contour that corresponds to the document
    result = []
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
        
        for k in key_gen:
            print(k)
        
        x,y,w,h = cv2.boundingRect(student_id_coor)
        student_id_img = image[y:y + h, x:x + w]
        
#         show_img(student_id_img)
        student_id_block = retr_student_id(student_id_img)

        print('---student id------')
        for num in student_id_block:
            print(num)

            
        x,y,w,h = cv2.boundingRect(code_test_coor)
        code_test_img = image[y:y + h, x:x + w]
#         show_img(code_test_img)
        
        code_id = retr_codetest_id(code_test_img)

        print('---code test id------')
        for num in code_id:
            print(num)
    
    result.append(key_gen)
    result.append(student_id_block)
    result.append(code_id)
    return result


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


# start_time = time.time()


# In[15]:


# for i in range(200):
#     filename = 'answer_gen/ans_' + str(i + 1) + '.png'
#     image = cv2.imread(filename)
    
#     print('---------------Scanning img ' + str(i + 1) + '--------------------------')
#     retrInfo(image)


# In[16]:


# print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:





# In[ ]:





# In[ ]:





# # API 

# In[170]:


from flask import Flask, request, redirect, jsonify

from functools import wraps
from flask_restful import Resource, Api, reqparse
import werkzeug
from werkzeug.utils import secure_filename
import pandas as pd
import ast

app = Flask(__name__, static_url_path="/static")
api = Api(app)




# In[171]:


# app.url_map


# In[172]:


class UploadImage(Resource):
    def post(self):
        result = []
        parse = reqparse.RequestParser()
        parse.add_argument('image', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        image_file = args.get("image")
        
        if image_file is not None:
            filename = secure_filename(image_file.filename)
            image_file.save(filename)
            image = cv2.imread(filename)
            result = retrInfo(image)
        return result


# In[173]:


api.add_resource(UploadImage, '/api')


# In[174]:

if __name__ == "__main__":
  app.run()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




