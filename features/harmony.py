#harmony calculation: written initially by Nicole and Lettie
import math
import numpy as np
from scipy.signal import argrelextrema

#HUE = 0 - 180 (179?)
#SATURATION = 0 - 255
#VALUE = 0 - 255

#TODO: Does this need to be generalized? right now it is specific to hue
#if so the '22.5' needs to be changed to the range of the new metric / 8
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



