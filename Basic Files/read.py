from types import FrameType
import cv2 as cv

# img = cv.imread('Resources/Photos/cat.jpg') #reading in a simple local cat file

# cv.imshow('Cat', img)

def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

# cv.waitKey(0) #wait infinitely for a key to be pressed

capture = cv.VideoCapture('Resources/Videos/dog.mp4')

while True:
     isTrue, frame = capture.read() #capture the video frame by frame
     
     frame_resized = rescaleFrame(frame, 0.2)
     
     cv.imshow('Video Resized', frame_resized)
     
     if cv.waitKey(20) & 0xFF==ord('d'): #if the letter d is pressed, break the loop and stop displaying    
        break

capture.release()
cv.destroyAllWindows()
