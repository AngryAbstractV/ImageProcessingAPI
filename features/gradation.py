import time

import cv2
import numpy as np

"""
    Gradation is based on change of colors or density throughout an image. The program uses Sobel Filter, 
    already provided by OpenCV. We take the x, y sobel filter scores and create a magnitude of both scores.
    We take the first channel from the magnitude score and modulo it until it matches the other scores.
    We then take the inverse because Sobel detects edges, which is the opposite of gradation
    - Daniel Martinez -
"""


def calc_pixel_change(matrix):
    x, y, z = matrix.shape
    vertical_score = 0
    horizontal_score = 0
    for i in range(x):
        horizontal_score += np.abs(np.diff(matrix[i, i, :], n=x - 1))[0]  # Don't ask me what this does
        vertical_score += np.abs(np.diff(matrix[:, i, i], n=y - 1))[0]  # I will cry if you do
    return vertical_score, horizontal_score


def calcGradation(curr_hsv_img):
    gX = cv2.Sobel(curr_hsv_img, cv2.CV_64F, 1, 0)  # Horizontal Sobel Scan
    gY = cv2.Sobel(curr_hsv_img, cv2.CV_64F, 0, 1)  # Vertical Sobel Scan
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    magnitudeScore = cv2.sumElems(magnitude)  # Sum
    magnitudeScore = magnitudeScore[0]

    while magnitudeScore >= 1.0:
        magnitudeScore /= 10

    if magnitudeScore == 0:
        return 0
    else:
        return 1 - magnitudeScore

