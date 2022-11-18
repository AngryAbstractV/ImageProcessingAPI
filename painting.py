import cv2
import numpy as np
# from features.balance import calcBalance
# from features.emphasis import calcEmphasis
# from features.movement import calcMovement
from features.gradation import calcGradation
from features.harmony import calcHarmony
from features.variety import calcVariety


# from gradation import calcGradation
# from movement import calcMovement

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
        self.properties_list[4] = calcGradation(self.hsv_img)

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