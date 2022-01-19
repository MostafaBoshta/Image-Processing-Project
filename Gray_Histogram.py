# -*- coding: utf-8 -*-
"""
Created on Fri May 21 18:20:39 2021

@author: MOUSE10
"""
"""
Histograms: compute the pixel intensity distribution in an image in terms of graphes.
"""

#pylint:disable=no-member

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('Images/cats.jpg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#GRayscale histogram
"""
For None: of there was no mask,it is written none and if there was a mask, the name of
the mask is written.
The firt arguement: is a list of image you want to pas to compute histoframs for.
The second arguement: is the number of channels, for RGB it is 3.
The fourth arguement: is the number of pins to compute the histogram.
The fifth arguement: is the range for pixel intensities.
"""
gray_hist = cv.calcHist([gray], [0], None, [256], [0,256] )

plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()
cv.waitKey(0)