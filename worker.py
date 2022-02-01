#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
#from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2
import os
import time
import string
import random
import multiprocessing as mp
from keras.models import load_model

model = load_model('weight.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[ ]:





# In[ ]:





# In[2]:


font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (255, 0, 0)
lineType = 1


# In[ ]:





# In[3]:


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


# In[4]:


def resize_img_data(img):
    img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
    img = img.reshape((28, 28, 1))
    img = np.reshape(img,[1,28,28,1])
    return img


# In[ ]:





# In[5]:


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


# In[6]:


def retr_codetest_id(code_test_img):
    
    code_id = {}
    
    legal_filled = 1
    
    code_test_img = cv2.cvtColor(code_test_img, cv2.COLOR_BGR2GRAY)
    code_test_img = cv2.threshold(code_test_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    code_test_cols = np.array_split(code_test_img, 3, 1)

#     show_img(code_test_img)


    for i, cols in enumerate(code_test_cols):
#         show_img(cols)

        
#         model predict
        filled_idx = []
        choice = []
        
        code_id_bubble = np.array_split(cols, 10, 0)
        for j, b in enumerate(code_id_bubble):
            


            temp_bb = resize_img_data(b.copy())
            classes = model.predict_on_batch(temp_bb)
            classes = list(classes[0])
            
            if(classes[1] > 0.9):
                filled_idx.append(j)
#             else:
#                 print('no choice code id')
#                 print(i + 1 , j + 1)
#                 show_img(b)
        
        if len(filled_idx) > legal_filled:
            choice = 'illegal'
        elif len(filled_idx) == 0:
            choice = "blank"
        else:
            choice = filled_idx
            
        code_id[str(i + 1)] = choice
        
#         print('code id filled : ' + str(choice))

        
    
#         old method
#         max_nonz = 0
#         filled_idx = -1
#         code_id_bubble = np.array_split(cols, 10, 0)

#         for j, b in enumerate(code_id_bubble):

#             curr_nonz = cv2.countNonZero(b)
# #             show_img(b)

#             print((i + 1, j + 1, curr_nonz))

#             if curr_nonz > max_nonz:
#                 filled_idx = j
#                 max_nonz = curr_nonz    
#         code_id[str(i + 1)] = filled_idx
        
# #         print('code_id id filled : ' + str(filled_idx))
        
    return code_id


# In[7]:


def retr_student_id(student_id_img):
 
    student_id = {}
    
    legal_filled = 1
    
    
    student_id_img = cv2.cvtColor(student_id_img, cv2.COLOR_BGR2GRAY)
    student_id_img = cv2.threshold(student_id_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
#     show_img(student_id_img)

    student_id_cols = np.array_split(student_id_img, 6, 1)
    
    for i, cols in enumerate(student_id_cols):

        
        student_id_bubble = np.array_split(cols, 10, 0)

        
#         model predict
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
#         print('student id filled : ' + str(choice))



#             old method

#         max_nonz = 0
#         filled_idx = -1
        
#         student_id_bubble = np.array_split(cols, 10, 0)
#         for j, b in enumerate(student_id_bubble):
#             curr_nonz = cv2.countNonZero(b)
                
#             print((i + 1, j + 1, curr_nonz))
# #             show_img(b)

#             if curr_nonz > max_nonz:
#                 filled_idx = j
#                 max_nonz = curr_nonz    
#         student_id[str(i + 1)] = filled_idx
        
#         print('student id filled : ' + str(filled_idx))
#         show_img(student_id_bubble[filled_idx])

    return student_id


# In[ ]:





# In[8]:


def get_choice(argument):
    switcher = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
    }
    return switcher.get(argument, "blank")


# In[9]:


def get_answer(answer_list):
    
    key_gen = {}
    
    legal_filled = 2
    
    for i, ans in enumerate(answer_list):
        ans = cv2.cvtColor(ans, cv2.COLOR_BGR2GRAY)
        ans = cv2.threshold(ans, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
#         show_img(ans)
        
        bubble_row = np.array_split(ans, 5, 1)
        bubble_row.pop(0)
        
        
#         model predict

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
        
#         print('answer id filled : ' + str(choice))

               
    
# cách 2: tìm lựa chọn theo giá trị lớn nhất

#         arr_choice = []
#         max_nonz = 0
#         choice = ''
#         filled_idx = -1
#         for j, b in enumerate(bubble_row):
#             curr_nonz = cv2.countNonZero(b)
            
#             print((i + 1, j + 1, curr_nonz))
            
# #             show_img(b)
            
#             if curr_nonz > max_nonz:
#                 filled_idx = j
#                 max_nonz = curr_nonz
                
#         choice = get_choice(filled_idx)
    
#         arr_choice.append(choice)
#         key_gen[str(i + 1)] = arr_choice
    return key_gen


# In[10]:


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





# In[11]:


def retrInfo(org_img):
    result = {}
    
#     org_img = cv2.imread(filename)

    # find contours in the edge map, then initialize
    # the contour that corresponds to the document

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
        
#         for i in key_gen:
#             print(i, key_gen[i])
#         print(key_gen)
        
        x,y,w,h = cv2.boundingRect(student_id_coor)
        student_id_img = image[y:y + h, x:x + w]
        
#         show_img(student_id_img)
        student_id_block = retr_student_id(student_id_img)

#         print('---student id------')
#         for i in student_id_block:
#             print(i, student_id_block[i])
#         print(student_id_block)

            
        x,y,w,h = cv2.boundingRect(code_test_coor)
        code_test_img = image[y:y + h, x:x + w]
#         show_img(code_test_img)
        
        code_id = retr_codetest_id(code_test_img)

#         print('---code test id------')
#         for i in code_id:
#             print(i, code_id[i])


        result['answer'] = key_gen
        result['student_id'] = student_id_block
        result['code_id'] = code_id
        
    return result


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


# start_time = time.time()


# In[13]:



# for i in range(1):
#     filename = 'answer_gen/ans_' + str(i + 1) + '.png'

#     filename = 'pic/omr_org_cp3.png'
#     filename = 'pic/omr_org_cp4.png'
#     filename = 'pic/omr_test_03.png'
#     filename = 'pic/omr_org.png'

#     image = cv2.imread(filename)
#     print('---------------Scanning img ' + str(i + 1) + '--------------------------')
#     retrInfo(image)


# In[14]:


# pool = mp.Pool(processes=2)
# results = [pool.apply_async(retrInfo, args=
#                                 ('answer_gen/ans_' + str(i + 1) + '.png',)) for i in range(5)] # maps function to iterator
# output = [p.get() for p in results]   # collects and returns the results
# for r in output:
#     print(r)   # read tuple elements


# In[15]:


# print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:





# In[16]:





# In[ ]:




# In[ ]:





# In[ ]:




