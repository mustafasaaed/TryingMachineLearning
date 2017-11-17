import numpy as np
import cv2

img = cv2.imread('..\Images\watch.jpg', cv2.IMREAD_COLOR)

cv2.line(img, (0, 0), (150, 150), (0, 255, 0), 15)
cv2.rectangle(img, (15, 25), (200, 200), (0, 0, 255), 8)
cv2.circle(img, (150, 63), 50, (0, 0, 0), -1)

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# pts = pts.reshape((-1,1,2))
cv2.polylines(img, [pts], True, (255,255,36), 5)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'open Cv !', (0,130), font, 1, (255,0,0), 2,cv2.LINE_AA)

cv2.imshow('images', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
