import numpy as np
import cv2
points_list = []

# Function to load an image
def load_image(path_img):
    return cv2.imread(path_img)

# Function to zoom into the image
def zoom(img, zoom_factor=1.5):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

# Function to crop the zoomed image
def crop_image(img, zoom_factor=3):
    frame_height, frame_width = img.shape[:2]
    zoomed_frame = zoom(img, zoom_factor)

    zoomed_height, zoomed_width = zoomed_frame.shape[:2]
    crop_x = (zoomed_width - frame_width) // 2
    crop_y = (zoomed_height - frame_height) // 2 - 500

    # Ensure cropping coordinates are within valid bounds
    crop_x = max(crop_x, 0)
    crop_y = max(crop_y, 0)
    end_x = min(crop_x + frame_width, zoomed_width)
    end_y = min(crop_y + frame_height, zoomed_height)

    cropped_frame = zoomed_frame[crop_y:end_y, crop_x:end_x]

    if cropped_frame.size == 0:
        raise ValueError("Cropped frame is empty. Check the cropping coordinates and zoom factor.")

    return cropped_frame

# Main code
path_img = 'pin/Pindetect_Test.jpg'
img = load_image(path_img)
if img is None:
    raise ValueError("Error: Could not open the image.")

# Zoom and crop the image before any processing
zoom_factor = 2  # Adjust the zoom factor as needed
cropped_frame = crop_image(img, zoom_factor)

# Convert BGR to HSV
hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

# Define range of white color in HSV
lower_white = np.array([28, 0, 50]) 
upper_white = np.array([180, 55, 255])

# Threshold the HSV image to get only white colors
mask = cv2.inRange(hsv, lower_white, upper_white)

# Find contours
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for index, c in enumerate(contours):
    rect = cv2.boundingRect(c)
    x, y, w, h = rect
    area = w * h

    epsilon = 0.08 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    if area > 200000:
        # cv2.drawContours(cropped_frame, [approx], -1, (0, 0, 255), 5)
        # cv2.rectangle(cropped_frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
        # print('approx', approx)
        for point in approx:
            points_list.append((point[0][0], point[0][1]))
            # cv2.circle(cropped_frame, (point[0][0], point[0][1]), 30, (0, 0, 255), -1)

# Process the image to detect pins and edges
print(points_list)

for index, point in enumerate(points_list):
    if (index == 1):
        cv2.circle(cropped_frame, point, 15, (0, 0, 255), -1)
    
# Save the result
output_path = '/mnt/data/output.png'
cv2.imwrite(output_path, cropped_frame)

# Display the images
cv2.namedWindow('Cropped Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Cropped Image', 1300, 700)  # Resize window to 600x700 pixels
cv2.imshow('Cropped Image', cropped_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()