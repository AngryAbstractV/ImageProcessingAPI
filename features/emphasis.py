import cv2
import numpy as np
import math
# NOTE: opencv must be installed with the 'contrib' extra modules installed.
# > pip install opencv-python
# > pip install opencv-contrib-python

def genSaliencyMap(image):
    # returns an image that is the saliency map
    # https://pyimagesearch.com/2018/07/16/opencv-saliency-detection/

    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    return saliencyMap


def genMask(image):
    #enumerate different mask equations
    img_wid = image.shape[1]
    img_len = image.shape[0]
    x = img_wid / 3
    y = img_len / 3

    # generate diagonal mask from left to right
    left_right_mask = np.zeros(shape=(img_len, img_wid, image.shape[2]), dtype=np.uint8)
    angle = math.atan(y/x) # arctangent
    thickness = int(2 * x * math.sin(angle)) # sine
    left_right_mask = cv2.line(left_right_mask, (0, 0), (img_wid, img_len), (255,250,255), thickness)
    left_right_mask = cv2.cvtColor(left_right_mask, cv2.COLOR_BGR2GRAY)
    ret, left_right_mask = cv2.threshold(left_right_mask, 200, 1, cv2.THRESH_BINARY)

    # generate diagonal mask from right to left
    right_left_mask = np.flipud(left_right_mask)

    # generate circular mask
    circle_mask = np.zeros(shape=(img_len, img_wid, image.shape[2]), dtype=np.uint8)
    radius = min(img_wid // 3, img_len // 3)
    center = (img_wid // 2, img_len // 2)
    circle_mask = cv2.circle(circle_mask, center, radius, (255,250,255), thickness=-1)
    circle_mask = cv2.cvtColor(circle_mask, cv2.COLOR_BGR2GRAY)
    ret, circle_mask = cv2.threshold(circle_mask, 200, 1, cv2.THRESH_BINARY)

    return (left_right_mask, right_left_mask, circle_mask)


def rfa(image_matrix):
    sum_saliency = 0
    sum_saliency_lr = 0
    sum_saliency_rl = 0
    sum_saliency_cir = 0
    masks = genMask(image_matrix)
    saliency = genSaliencyMap(image_matrix)
    lr_sal = saliency * masks[0]
    rl_sal = saliency * masks[1]
    cir_sal = saliency * masks[2]

    for x in range(image_matrix.shape[0]):
        for y in range(image_matrix.shape[1]):
            sum_saliency += saliency.item((x, y))
            sum_saliency_lr += lr_sal.item((x, y))
            sum_saliency_rl += rl_sal.item((x, y))
            sum_saliency_cir += cir_sal.item((x, y))

    if sum_saliency == 0:
        return 0

    max_sal = max(sum_saliency_cir, sum_saliency_lr, sum_saliency_rl)
        
    return (max_sal / sum_saliency)

def itten_color(image_matrix, eq_type):
    # various itten eqs
    # calculate standard deviation
    itten_color_scores = []
    return itten_color_scores

def calcEmphasis(image_matrix):
    img = cv2.cvtColor(image_matrix, cv2.COLOR_HSV2BGR)

    # calculate itten color scores

    # calculate RFA
    rfaScore = rfa(img)

    # calulate emphasis by combining itten and RFA
    emphasis_score = rfaScore
    return emphasis_score 

