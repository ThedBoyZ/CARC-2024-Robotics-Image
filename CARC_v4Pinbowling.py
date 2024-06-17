import cv2
import numpy as np

def zoom(img, zoom_factor=1.5):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

def refine_pin_detection(img, roi=None):
    frame_width, frame_height = img.shape[1], img.shape[0]

    # Zoom into the frame
    zoom_factor = 3
    zoomed_frame = zoom(img, zoom_factor)

    # Calculate the coordinates for cropping the center of the zoomed image
    zoomed_height, zoomed_width = zoomed_frame.shape[:2]
    crop_x = (zoomed_width - frame_width) // 2
    crop_y = ((zoomed_height - frame_height) // 2) - 200

    # Crop the zoomed frame to the original frame size centered
    cropped_frame = zoomed_frame[crop_y:crop_y+frame_height, crop_x:crop_x+frame_width]

    # Pre-process the Frame
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Invert the grayscale image
    gray_inverted = cv2.bitwise_not(blur)

    # Create a binary thresholded image
    _, binary = cv2.threshold(gray_inverted, 97, 255, cv2.THRESH_BINARY)

    # Use morphological operations to clean up the binary image
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)

    # Edge Detection for Field Boundaries
    edges = cv2.Canny(binary, 40, 150)

    # Find Contours for Field Boundaries
    contours_field, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find Contours for Pins
    contours_pins, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw Contours on the Frame
    contour_frame = cropped_frame.copy()
    cv2.drawContours(contour_frame, contours_field, -1, (0, 255, 0), 2)
    cv2.drawContours(contour_frame, contours_pins, -1, (0, 0, 255), 2)
    
    # Draw the bounding box on the original image
    x, y, w, h = 50, 50, 1500, 700  # Adjusted coordinates, change if necessary
    cv2.rectangle(cropped_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Filter the contours to identify standing pins
    pin_contours = []
    for contour in contours_pins:
        area = cv2.contourArea(contour)
        # Filter based on area and aspect ratio to detect pin heads
        if 0 < area < 1600:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / float(w)
            if 0.8 < aspect_ratio < 5:
                pin_contours.append(contour)

    num_pins = len(pin_contours)

    cv2.putText(contour_frame, f'Pins: {num_pins}', (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return contour_frame, num_pins

# def count_green_objects(img, roi=None):
#     if roi:
#         x, y, w, h = roi
#         img = img[y:y+h, x:x+w]
    
#     # Convert the image to HSV color space
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     # Define the green color range in HSV
#     lower_green = np.array([35, 40, 40])
#     upper_green = np.array([85, 255, 255])

#     # Create a mask for green color
#     mask = cv2.inRange(hsv, lower_green, upper_green)

#     # Find contours of the green objects
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Filter out small contours
#     green_objects = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

#     return len(green_objects)

# Load the image
img_path = 'pin/Pindetect_Test.jpg'
image = cv2.imread(img_path)
if image is None:
    print("Error: Could not open image.")
    exit()

# # Define the bounding box (ROI) coordinates
# roi_x, roi_y, roi_w, roi_h = 50, 50, 1500, 700  # Adjusted coordinates, change if necessary
# roi = (roi_x, roi_y, roi_w, roi_h)

# Apply the refined pin detection with ROI
refined_contour_frame, refined_num_pins = refine_pin_detection(image)

# # Count the green objects within the ROI
# num_green_objects = count_green_objects(image, roi)

cv2.namedWindow('Refined Contours', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Refined Contours', 1300, 600)  # Resize window to 1300x600 pixels
# Display the result
cv2.imshow('Refined Contours', refined_contour_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
output_path = '/mnt/data/RefinedContourPin_image.png'
cv2.imwrite(output_path, refined_contour_frame)
print(f"Processing complete. Image saved to {output_path}. Pins detected: {refined_num_pins}.")
