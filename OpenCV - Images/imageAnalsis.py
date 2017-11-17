import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('Images\watch.jpg',cv2.IMREAD_GRAYSCALE)

# Open CV

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Saveing the image
cv2.imwrite('Images\grayImage.png', img)

#matlap

# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.plot([50,100],[80,100], 'r', linewidth=5)
# plt.show()

