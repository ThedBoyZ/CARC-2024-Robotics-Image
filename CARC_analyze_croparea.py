import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('9.jpg')                   # Read image
img = cv2.resize(img, (672, 672))                    # Resize image     
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_gray = np.array([0, 0, 0], np.uint8)
upper_gray = np.array([0, 0, 45], np.uint8)
mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

start_point = (0,0)
end_point = (675,200)
color= (0, 0, 0)
# Line thickness 
thickness = -1
image1 = cv2.rectangle(mask_gray, start_point, end_point, color, thickness) 

image = image1

img_res = cv2.bitwise_and(img, img, mask = mask_gray)
cv2.imshow('Pindetect Test', mask_gray)
cv2.imwrite('5.jpg',mask_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()