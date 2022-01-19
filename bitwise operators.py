# -*- coding: utf-8 -*-
"""
Created on Fri May 21 21:34:37 2021

"""

#pylint:disable=no-member

import opencv as cv
import numpy as np

"""
They are usually used alotespecally with masking.
"""
blank = np.zeros((400,400), dtype='uint8')

"""
(30, 30): is the start point and go all the way across till(370, 370).
255 is color white, -1 is to fill the shape.
"""
rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, 0)
circle = cv.circle(blank.copy(), (200,200), 200, 255, -1)

cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)

# bitwise AND --> intersecting regions
bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow('Bitwise AND', bitwise_and)

# bitwise OR --> non-intersecting and intersecting regions
bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow('Bitwise OR', bitwise_or)

# bitwise XOR --> non-intersecting regions
bitwise_xor = cv.bitwise_xor(rectangle, circle)
cv.imshow('Bitwise XOR', bitwise_xor)

# bitwise NOT
bitwise_not = cv.bitwise_not(circle)
cv.imshow('Circle NOT', bitwise_not)

cv.waitKey(0)