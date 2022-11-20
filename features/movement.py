#movement calculation: written initially by Nicole and Lettie
import math
import numpy as np
import cv2

#HUE = 0 - 180 (179?)
#SATURATION = 0 - 255
#VALUE = 0 - 255

def displayPath(img, salientList, mapLength):
    for x in range(mapLength - 1):
        start = salientList[x]
        end = salientList[x+1]
        # draw a line starting at [x] and ending at [x+1]
        if x % 9 == 0:
            img = cv2.line(img,(start[1], start[0]), (end[1], end[0]), (0, 255, 0), 1)
        # img = cv2.line(img, (start[1], start[0]), (end[1], end[0]), (0, 255, 0), 1)
    return img

def genSaliencyMap(image):
    # returns an image that is the saliency map
    # https://pyimagesearch.com/2018/07/16/opencv-saliency-detection/

    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    return saliencyMap


def genAvgSacadeLength(saliencyMap, mapLength):
    salientList = np.argpartition(saliencyMap, saliencyMap.size - mapLength, axis=None)[-mapLength:]
    salientList = np.column_stack(np.unravel_index(salientList, saliencyMap.shape))

    sacades = 0
    newLen = mapLength - 1

    for x in range(mapLength - 1):
        # √[(x₂ - x₁)² + (y₂ - y₁)²] 
        distance = math.sqrt(abs((salientList[x][0] - salientList[x+1][0]) ** 2 + (salientList[x][1] - salientList[x+1][1]) ** 2))
        if distance > 20:
            sacades += distance
        else:
            newLen = newLen - 1

    return (sacades / newLen)


def calcMovement(img):
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    salienceMap = genSaliencyMap(img)
    standardDev = np.std(salienceMap)
    maplength = int(standardDev * salienceMap.shape[0] * salienceMap.shape[1])

    if standardDev == 0:
        return 0

    avgSacadeLength = genAvgSacadeLength(salienceMap, maplength)
    maxAvgLen = (math.sqrt((img.shape[0]) ** 2 + (img.shape[1]) ** 2)) / 2

    movement = ((avgSacadeLength / maxAvgLen))

    if movement > 1.0:
        movement = 1.0

    return movement