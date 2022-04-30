import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    ret_val, img = cam.read()
    cv2.imshow('cam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('p'):
        cv2.imwrite("example.jpg",img)
cam.release()
cv2.destroyAllWindows()