# -*- coding: utf-8 -*-
"""Variety

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hNIEO5TCZU-j63LMYGV06nAytVyfv2_e
"""

'''
@author Sinh Mai and Thu Thach
@version 1.5
@date 11/2/2022
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

"""
Counts and sorts pixels based on RGB colors.
Returns an array that contains the counts of each color.
"""

def get_hist(img):
  counts, bins = np.histogram(img,bins=11)
  return counts


"""
Calculate a value to send to ML model
scale of 0-1
A score of 1 means that distribution of colors is uniform
A score of 0 means that distribution of colors is not uniform
"""

def calcVariance(avg, count, total):
    sqSum = 0
    for i in range(len(count)):
        sqSum += (count[i] - avg)**2
    return sqSum/(len(count))

def maxVariance(avg, count, total):
    i = len(count)
    m = (i-1)*(avg)**2 + (total - avg)**2
    return m/i

def calcScore(count, total):
    avg = total / len(count)
    return 1 - (calcVariance(avg,count,total) / maxVariance(avg,count,total))

"""
Main function for the driver to use.
"""
def calcVariety(img):
    #img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
    count = get_hist(img)
    total = np.sum(count)
    return calcScore(count, total)
