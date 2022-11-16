from fastapi import FastAPI, File, UploadFile
import numpy as np
import random
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import cv2
from scipy.signal import argrelextrema
import math
import tensorflow as tf

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
]

app = FastAPI(middleware=middleware)


@app.get("/")
async def main():
    return {"Scores": [random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100,
                       random.randint(10, 100) / 100, random.randint(10, 100) / 100]}


@app.post("/upload")
async def root(file: UploadFile = File(...)):
    contents = await file.read()
    properties_list = [0] * 6
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    painting = Painting(img)
    painting.preprocessing()
    painting.calculateProperties()
    # processedIMG = preprocessing(img)
    # properties_list[2] = calcHarmony(processedIMG)
    # properties_list[3] = calcVariety(processedIMG)
    return painting.properties_list

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    prediction = pred(img)
    prediction = np.array(prediction)
    return str(prediction)


model = tf.keras.models.load_model('model_03_a_0.2411_l_1.9252_va_0.2131_vl_1.9761.h5')
def pred(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (120, 120),interpolation=cv2.INTER_CUBIC)
    img = np.float32(img)
    img_array = tf.expand_dims(img, 0)
    predictions = model.predict(img_array)
    return predictions[0]


class Painting:

    # init method or constructor
    def __init__(self, img):
        # self.name = name
        # self.imageAddress = imageAddress
        self.properties_list = [0] * 6
        self.img = img
        self.hsv_img = np.zeros((1, 1, 1), dtype=np.int32)

    def preprocessing(self):
        # pre-processesing on image

        # input image
        # self.img = cv2.imread(self.imageAddress, 1)

        self.hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        # hsv_img is a 3d numpy array with the inner array having values:
        # HUE = 0 - 180 (179?)
        # SATURATION = 0 - 255
        # VALUE = 0 - 255

        # Scale image, preserving aspect ratio
        pix_threshold = 500
        img_wid = self.hsv_img.shape[1]
        img_len = self.hsv_img.shape[0]
        new_wid = 0
        new_len = 0

        if img_wid > pix_threshold or img_len > pix_threshold:
            # find out which is bigger
            if img_wid > img_len:
                new_wid = pix_threshold
                scale = (new_wid * 100) // img_wid
                new_len = (img_len * scale) // 100
            elif img_len > img_wid:
                new_len = pix_threshold
                scale = (new_len * 100) // img_len
                new_wid = (img_wid * scale) // 100
            else:
                # wid and len are equal
                new_wid = pix_threshold
                new_len = pix_threshold

            dim = (new_wid, new_len)
            self.hsv_img = cv2.resize(self.hsv_img, dim, interpolation=cv2.INTER_AREA)

        print("preprocessing complete")
        return

    def calculateProperties(self):
        # call all feature algorithms and return a list of feature scores normalized from 0 to 1

        # TODO: remove once a feature is complete and integrated into driver
        # self.properties_list = [0.5]*6

        # call balance
        # self.properties_list[0] = calcBalance(self.hsv_img)

        # call emphasis
        # self.properties_list[1] = calcEmphasis(self.img)

        # call harmony
        self.properties_list[2] = calcHarmony(self.hsv_img)

        # call variety
        self.properties_list[3] = calcVariety(self.hsv_img)

        # call gradation
        # self.properties_list[4] = calcGradation(self.hsv_img)

        # call movement
        # self.properties_list[5] = calcMovement(self.hsv_img)
        print("properties calculations complete")
        return

    def getImage(self):
        return self.img

    def getHSVImage(self):
        return self.hsv_img

    def getPropertiesList(self):
        return self.properties_list


def genNeighborhoodHistogram(neighborhoodMatrix, setting='hue'):
    neighborhoodHistogram = [0]*8
    shape = neighborhoodMatrix.shape
    # for each pixel in neighborhood:
    for x in range(shape[0]):
        for y in range(shape[1]):

            # grab first number in tuple at (x, y) which is the hue, determine which 'bucket' it belongs to
            if setting == 'hue':
                val = int(neighborhoodMatrix.item((x, y, 0)) // 22.5)
            elif setting == 'saturation':
                val = int(neighborhoodMatrix.item((x, y, 1)) // 32)
            elif setting == "value":
                val = int(neighborhoodMatrix.item((x, y, 2)) // 32)

            # increment count for bucket
            neighborhoodHistogram[val] += 1
    return neighborhoodHistogram

# calc max modes OPT 1
# check edge cases for histogram
# calculate local maxima and minima
# partition histogram array into c and I\c 
# determine value (count) of maximum, and index of location

def calcPixelHarmony(neighborhoodHistogram):
    modes = [[0,0],[0,0]]
    #find the maxima and minima
    maxima = argrelextrema(np.array(neighborhoodHistogram), np.greater, mode= 'wrap')
    #print(maxima)
    #minima = argrelextrema(neighborhoodHistogram, np.less)
    modes[0][1] = max(neighborhoodHistogram)
    modes[0][0] = neighborhoodHistogram.index(modes[0][1])
    for i in maxima[0]:
        if i == modes[0][0]:
            continue
        secondMaxValue = neighborhoodHistogram[i]
        if secondMaxValue > modes[1][1]:
            modes[1][0], modes[1][1] = i, secondMaxValue

    return calcModeHarmony(modes)
"""
# calc max modes OPT 2
# consider all possible values of c (splitting histogram in all possible ways?)
# harmony intensity will calculated 56 times per pixel to find lowest possible value (why lowest??)
def calcPixelHarmony(neighborhoodHistogram):
    minHarmony = 1
    modes = [[0,0],[0,0]]
    for i in range(len(neighborhoodHistogram)-1):
        for j in range(i, len(neighborhoodHistogram)):
            if i%7 == j%7:
                continue
            c, ic = neighborhoodHistogram[i:j], neighborhoodHistogram[j:] + neighborhoodHistogram[:i]
            #print(f"i: {i} j: {j}")
            modes[0][1], modes[1][1] = max(c), max(ic)
            modes[0][0], modes[1][0] = c.index(modes[0][1]), ic.index(modes[1][1])
            pixelHarmony = calcModeHarmony(modes)
            if pixelHarmony < minHarmony:
                minHarmony = pixelHarmony
            modes[0], modes[1] = modes[1], modes[0]
            pixelHarmony = calcModeHarmony(modes)
            if pixelHarmony < minHarmony:
                minHarmony = pixelHarmony
    return minHarmony
"""

#calculate the individual pixel harmony based on a tuple of the two max modes 
#modes[colorcatagory, quantity]
def calcModeHarmony(modes):
    # checks if only 1 maxima was returned
    if modes[1][1] == 0:
        index_diff = 4
        value_diff = 0
    else: 
        index_diff = min((abs(modes[0][0] - modes[1][0])), 8 - (abs(modes[0][0] - modes[1][0])))
        value_diff = -abs(modes[0][1] - modes[1][1])
        #value_diff = (-abs(modes[0][1] - modes[1][1])) * ((modes[0][1] + modes[1][1]) / (dimension ** 2))
    pixelHarmony = math.exp(value_diff) * index_diff
    return pixelHarmony

#main function / driver of harmony calculation
def calcHarmony(hsvImg):
    # sum of each pixel's harmony intensity
    totalHarmony = 0
    neighborhood_dimension = 9 #grid must be odd!!
    xy_init = neighborhood_dimension // 2 # 4
    hueval = 0
    satval = 0
    valval = 0

    img_wid = hsvImg.shape[1] # left to right
    img_len = hsvImg.shape[0] # up to down


    # anchor coords represented by x, y. x goes lengthwise, y goes widthwise
    for x in range(xy_init, (img_len - xy_init)):  #img_len - xy_init should be 4 less than the end of line
        for y in range(xy_init, (img_wid - xy_init)):
            if x % 3 == 0 and y % 3 == 0:
            # pull out submatrix surrounding anchor
                neighMatrix = hsvImg[(x - xy_init):(x + xy_init) + 1, (y - xy_init):(y + xy_init) + 1]

                histogram = genNeighborhoodHistogram(neighMatrix, setting='hue')
                hueval += calcPixelHarmony(histogram)

                histogram = genNeighborhoodHistogram(neighMatrix, setting='saturation')
                satval += calcPixelHarmony(histogram)

                histogram = genNeighborhoodHistogram(neighMatrix, setting='value')
                valval += calcPixelHarmony(histogram)
        
    totalHarmony = (hueval + satval + valval) / 3
    
    scalingValue = ((img_len - (xy_init * 2)) * (img_wid - (xy_init * 2))) * 4
    totalHarmony = totalHarmony / scalingValue
    totalHarmony = totalHarmony * 10
    totalHarmony -= .1
    return totalHarmony



def get_hist(img):
    counts, bins = np.histogram(img, bins=11)
    return counts


def calcVariance(avg, count, total):
    sqSum = 0
    for i in range(len(count)):
        sqSum += (count[i] - avg) ** 2
    return sqSum / (len(count))


def maxVariance(avg, count, total):
    i = len(count)
    m = (i - 1) * (avg) ** 2 + (total - avg) ** 2
    return m / i


def calcScore(count, total):
    avg = total / len(count)
    return 1 - (calcVariance(avg, count, total) / maxVariance(avg, count,
                                                              total))


"""
Main function for the driver to use.
"""


def calcVariety(img):
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    count = get_hist(img)
    total = np.sum(count)
    return calcScore(count, total)
