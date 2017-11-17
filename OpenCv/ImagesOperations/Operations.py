import numpy as np
import cv2

img = cv2.imread('..\Images\watch.jpg', cv2.IMREAD_COLOR)


img[55,55] = [0,0,0]
# px = img[55, 55]

# roi = img[100:150, 100:150]
#print(roi)

# img[100:150, 100:150] = [0,0,0]


watch_face = img[37:111, 107:194]
img[0:74, 0:87] = watch_face

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


