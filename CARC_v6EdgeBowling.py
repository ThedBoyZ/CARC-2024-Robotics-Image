import numpy as np
import cv2

points_list = []
pin_center = []

def load_image(path_img):
    return cv2.imread(path_img)

def zoom(img, zoom_factor=1.5):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

def refine_pin_detection_adjusted(img):
    frame_width, frame_height = img.shape[1], img.shape[0]
    zoom_factor = 2
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
            for point in approx:
                points_list.append((point[0][0], point[0][1]))

    # Initialize contour_frame
    print(points_list)
    contour_frame = cropped_frame.copy()

    for index, point in enumerate(points_list):
        if index == 1:
            cv2.circle(contour_frame, point, 15, (0, 0, 255), -1)
            
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    gray_inverted = cv2.bitwise_not(blur)
    _, binary = cv2.threshold(gray_inverted, 100, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(gray_inverted, 115, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)

    edges = cv2.Canny(binary, 40, 180)
    contours_pins, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_field, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(contour_frame, contours_pins, -1, (0, 255, 0), 2)
    cv2.drawContours(contour_frame, contours_field, -1, (0, 0, 255), 2)

    # Adjust the coordinates of the bounding box
    x, y, w, h = cv2.boundingRect(contours_field[0])
    new_x = x + 10  # Adjusted to properly fit the pins  ---> - 50 px
    new_y = y - 180  # Adjusted to properly fit the pins ---> -300 px
    new_w = w + 1020  # Adjusted width ----> + 1620 px
    new_h = h + 50  # Adjusted height  ----> +   50 px
    
                                            # Crop y rectangle
    cv2.rectangle(contour_frame, (new_x+120, new_y-100), (new_x + new_w , new_y + new_h), (255, 0, 0), 2)
    
    pin_contours = []
    for contour in contours_pins:
        area = cv2.contourArea(contour)
        if 40 < area < 1600:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / float(w)
            if 0.8 < aspect_ratio < 5:
                pin_contours.append(contour)
                
                # Add a small red dot at the center of each pin contour
                center_x, center_y = x + w // 2, y + h // 2
                cv2.circle(contour_frame, (center_x, center_y), 5, (0, 0, 255), -1)
    
    # Filter pins inside the adjusted rectangle and exclude areas near the edges
    edge_buffer = 7  # Buffer distance from the edge
    filtered_pins = [
        contour for contour in pin_contours 
        if new_x + edge_buffer + 120 <= cv2.boundingRect(contour)[0] <= new_x + new_w - edge_buffer
        and new_y - edge_buffer - 100 <= cv2.boundingRect(contour)[1] <= new_y + new_h - edge_buffer
    ]
    num_pins = len(filtered_pins)

    cv2.putText(contour_frame, f'Pins: {num_pins}', (25, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
    
    for contour in filtered_pins:
        x, y, w, h = cv2.boundingRect(contour)
        center_x, center_y = x + w // 2, y + h // 2
        pin_center.append((center_x, center_y))
    print(pin_center)
    # print(pin_center[0])    
      
    # Calculate distance between the first two points
    if len(pin_center) != 0:
        if len(points_list) >= 2:
            pt1 = points_list[1]
            pt2 = pin_center[6]
            distance = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            cv2.line(contour_frame, pt1, pt2, (255, 0, 0), 2)
            cv2.putText(contour_frame, f'Distance: {distance:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
        if distance >= 170 and distance <= 190:
            cv2.putText(contour_frame, f'A', (1800, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
            
    return contour_frame, num_pins
    
# new_image_path = 'pin/dataset_carc/A/in_calibrateline/image0.jpg' 
new_image_path = 'pin/dataset_carc/Pindetect_Test.jpg' 
new_image = load_image(new_image_path)
if new_image is None:
    raise ValueError("Error: Could not open the newly uploaded image.")

# Process the new image to detect pins
adjusted_contour_frame, adjusted_num_pins = refine_pin_detection_adjusted(new_image)
cv2.namedWindow('Refined Contours', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Refined Contours', 1300, 600)  # Resize window to 1300x600 pixels

# Display the result
cv2.imshow('Refined Contours', adjusted_contour_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
adjusted_output_path = '/mnt/data/AdjustedRefinedContourPin_image.png'
cv2.imwrite(adjusted_output_path, adjusted_contour_frame)

adjusted_output_path, adjusted_num_pins
