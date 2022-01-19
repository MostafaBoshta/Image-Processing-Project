# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 01:19:22 2021

@author: mosta
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def load_image(imagelocation):
    img = cv.imread(imagelocation)
    cv.imshow('Picture', img)
    
imageSelected = load_image('\Desktop/121104431_3321639724593155_5871112679046826586_n.jpg')   
#convert
def gray_color(imageSelected):
     gray = cv.cvtColor(imageSelected, cv.COLOR_BGR2GRAY)
     cv.imshow('Gray', gray)
     
gray_img = gray_color(imageSelected)     


#noise
def saltandpepper(imageSelected):
    img = cv.imread(imageSelected, 1)
    img = cv.cvtColor(imageSelected, cv.COLOR_BGR2RGB)
    plt.imshow(img)


#point transform ops
def colored_histogram(imageSelected):
    blank = np.zeros(imageSelected.shape[:2], dtype='uint8')
    mask = cv.circle(blank, (imageSelected.shape[1]//2,imageSelected.shape[0]//2), 100, 255, -1)

    masked = cv.bitwise_and(imageSelected,imageSelected,mask=mask)
    cv.imshow('Mask', masked)
    plt.figure()
    plt.title('Colour Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    colors = ('b', 'g', 'r')
    for i,col in enumerate(colors):
        hist = cv.calcHist([imageSelected], [i], mask, [256], [0,256])
        plt.plot(hist, color=col)
        plt.xlim([0,256])

    plt.show()

    cv.waitKey(0)
    
    
def gray_histogram(gray_img):
    gray_hist = cv.calcHist([gray_img], [0], None, [256], [0,256] )

    plt.figure()
    plt.title('Grayscale Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    plt.plot(gray_hist)
    plt.xlim([0,256])
    plt.show()
    cv.waitKey(0)
    
    
def colored_histogram_equalisation(imageSelected):
    R, G, B = cv.split(imageSelected)
    
    output1_R = cv.equalizeHist(R)
    output1_G = cv.equalizeHist(G)
    output1_B = cv.equalizeHist(B)
    
    output1 = cv.merge((output1_R, output1_G, output1_B))

#    clahe = cv2.createCLAHE()
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    output2_R = clahe.apply(R)
    output2_G = clahe.apply(G)
    output2_B = clahe.apply(B)
    
    output2 = cv.merge((output2_R, output2_G, output2_B))


    output = [imageSelected, output1, output2]
    titles = ['Original Image', 'Adjusted Histogram', 'CLAHE']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(output[i])
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

def gray_histogram_equalisation(gray_img):
    R, G, B = cv.split(gray_img)
    
    output1_R = cv.equalizeHist(R)
    output1_G = cv.equalizeHist(G)
    output1_B = cv.equalizeHist(B)
    
    output1 = cv.merge((output1_R, output1_G, output1_B))

#    clahe = cv2.createCLAHE()
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    output2_R = clahe.apply(R)
    output2_G = clahe.apply(G)
    output2_B = clahe.apply(B)
    
    output2 = cv.merge((output2_R, output2_G, output2_B))


    output = [gray_img, output1, output2]
    titles = ['Original Image', 'Adjusted Histogram', 'CLAHE']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(output[i])
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()
#local transform
def low_pass_filter(imageSelected):
    image1_np=np.array(imageSelected)
    fft1 = fftpack.fftshift(fftpack.fft2(image1_np))
    x,y = image1_np.shape[0],image1_np.shape[1]
    e_x,e_y=50,50
    bbox=((x/2)-(e_x/2),(y/2)-(e_y/2),(x/2)+(e_x/2),(y/2)+(e_y/2))
    low_pass=Image.new("L",(image1_np.shape[0],image1_np.shape[1]),color=0)
    draw1=ImageDraw.Draw(low_pass)
    draw1.ellipse(bbox, fill=1)
    low_pass_np=np.array(low_pass)
    filtered=np.multiply(fft1,low_pass_np)
    ifft2 = np.real(fftpack.ifft2(fftpack.ifftshift(filtered)))
    ifft2 = np.maximum(0, np.minimum(ifft2, 255))
    cv.imshow('lowpassfilter' , ifft2)
    
    
    
    
def median_blur_filter(gray_image):
    median = cv.medianBlur(gray_img,5)

    plt.subplot(121),plt.imshow(gray_img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(median),plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()    
    
    
#edge detection filters    
def lablacian_filter(gray_img):
    lap = cv.Laplacian(gray_img, cv.CV_64F)
    lap = np.uint8(np.absolute(lap))
    cv.imshow('Laplacian', lap)    

def sobel_filter(gray_img):
    sobelx = cv.Sobel(gray_img, cv.CV_64F, 1, 0)
    sobely = cv.Sobel(gray_img, cv.CV_64F, 0, 1)
    combined_sobel = cv.bitwise_or(sobelx, sobely)

    cv.imshow('Sobel X', sobelx)
    cv.imshow('Sobel Y', sobely)
    cv.imshow('Combined Sobel', combined_sobel)
             
def gussian_filter(gray_img):
    blur = cv.GaussianBlur(gray_img,(5,5))

    plt.subplot(121),plt.imshow(gray_img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()



def canny_filter(gray_img):
    canny = cv.Canny(gray_img, 150, 175)
    cv.imshow('Canny', canny)
    cv.waitKey(0)    

#Global transform ops
def line_detection(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150,apertureSize = 3)

    lines = cv.HoughLines(edges,1,np.pi/180,200)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    
def circle_detection(img):
    img = cv.medianBlur(img,5)
    cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    cv.imshow('detected circles',cimg)
    
    
#Morphological ops
def erosion_img(imageSelected):
    kernel = np.ones((5,5),np.uint8)
    erosion = cv.erode(imageSelected,kernel,iterations = 1)

def dialation_img(imageSelected):
    kernel = np.ones((5,5),np.uint8)
    dilation = cv.dilate(imageSelected,kernel,iterations = 1)     
     
def opening_img(imageSelected):
    kernel = np.ones((5,5),np.uint8)
    opening = cv.morphologyEx(imageSelected, cv.MORPH_OPEN, kernel)    
     
def closing_img(imageSelected):
    kernel = np.ones((5,5),np.uint8)
    closing = cv.morphologyEx(imageSelected, cv.MORPH_CLOSE, kernel)
        

     
    

