import sys
import cv2

print (cv2.__version__)

gst = "http://192.168.0.29:8080/video"

cap = cv2.VideoCapture(gst)
if not cap.isOpened() :
    print("capture failed")
    exit()

ret,frame = cap.read()
while ret :
    cv2.imshow('frame',frame)
    ret,frame = cap.read()
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()