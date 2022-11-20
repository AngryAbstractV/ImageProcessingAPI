
'''
@author Sinh Mai
@version 0.1
@date 11/16/2022
'''

import numpy as np
import cv2

'''
Algorithm
1. Convert image to GreyScale.
2. For each direction: Vertical, Horizontal (Diagonals optional)
    a. Slice image in half.
    b. Flip one half and subtract it from the other half.
    c. Average the values of the difference matrix.
    c. Subtract the average from 1 .
    
Option 1    
    3. Average the values of the vertical and horizontal symmetries
    4. Output (1 - result of step 3) 
    
Option 2
    3. output the highest average of the two directions

Notes:
1 = perfect symmetry
0 = no symmetry
'''

def bilateralSymmetry(img):
    
    if (img.shape[0] % 2 == 1):
        img.resize(((img.shape[0] + 1), img.shape[1]),refcheck=False)
    if (img.shape[1] % 2 == 1):
        img.resize((img.shape[0], (img.shape[1] + 1)), refcheck=False)

    img_hSplit = np.split(img, 2, 0)
    img_vSplit = np.split(img, 2, 1)

    v_difference = abs(img_vSplit[0] - np.fliplr(img_vSplit[1]))
    v_symmetry = np.average(v_difference)

    h_difference = abs(img_hSplit[0] - np.flipud(img_hSplit[1]))
    h_symmetry = np.average(h_difference)
    
    return (v_symmetry, h_symmetry)

def calcBalance(img):
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255

    symmetry = bilateralSymmetry(img)

    #average of both horizontal and vertical symmetry
    val_average = np.average(symmetry)

    return ((1- val_average))
    
"""
#Main method for testing
if __name__=='__main__':
    
    testImg = "ExamplePaintings/10--frantisek-kupka_the-gallien-girl-1910.png"

    img = cv2.imread(testImg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    val = calcBalance(img)

    print(val)
"""