import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edges', canny)

# this is an alternative to the canny method for finding contours, if the intensity of a pixel is below 125
#it is set to 0, if it is above 255 it is set to white
# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# cv.imshow('Thresh', thresh)

#finding the contours, returns a python list of coordinates of contours (contours) and the hierarchical 
#representation of the contours, how they are stacked (hierarchies)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) #or use CHAIN_APPROX_NONE
print(f'{len(contours)} contour(s) found!')

cv.waitKey(0)