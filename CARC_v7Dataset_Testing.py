import numpy as np
import cv2

# Initialize the points lists
points_list = []
points_select = []
pin_center = []

def load_image(path_img):
    return cv2.imread(path_img)

def zoom(img, zoom_factor=1.5):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    if brightness != 0:
        beta = brightness
        image = cv2.add(image, beta)
    return image

def refine_pin_detection_adjusted(img):
    img = adjust_brightness_contrast(img, brightness=2, contrast=21)
    
    frame_width, frame_height = img.shape[1], img.shape[0]
    zoom_factor = 2
    zoomed_frame = zoom(img, zoom_factor)

    zoomed_height, zoomed_width = zoomed_frame.shape[:2]
    crop_x = (zoomed_width - frame_width) // 2 + 120
    crop_y = (zoomed_height - frame_height) // 2 + 50

    # Ensure cropping coordinates are within valid bounds
    crop_x = max(crop_x, 0)
    crop_y = max(crop_y, 0)
    end_x = min(crop_x + frame_width, zoomed_width)
    end_y = min(crop_y + frame_height, zoomed_height)

    cropped_frame = zoomed_frame[crop_y:end_y, crop_x:end_x]
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

    # Define range of white color in HSV
    lower_white = np.array([40, 10, 60])  
    upper_white = np.array([180, 55, 255])
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    lower_green = np.array([85, 90, 70])
    upper_green = np.array([115, 205, 255])

    # Threshold the HSV image to get only green colors
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours for green pins
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    green_pin_centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 0:
            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w // 2, y + h // 2
            green_pin_centers.append((center_x, center_y))
            cv2.circle(cropped_frame, (center_x, center_y), 2, (0, 255, 0), -1)  # Green dot for green pin center
    print(len(green_pin_centers))
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
        if area > 200:
            for point in approx:
                points_list.append((point[0][0], point[0][1]))

    # Initialize contour_frame
    contour_frame = cropped_frame.copy()
            
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)

    gray_inverted = cv2.bitwise_not(blur)
    _, binary = cv2.threshold(gray_inverted, 37, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(gray_inverted, 115, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)

    edges = cv2.Canny(binary, 40, 210)
    contours_pins, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_field, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(contour_frame, contours_pins, -1, (0, 255, 0), 1)

    # Adjust the coordinates of the bounding box
    x, y, w, h = cv2.boundingRect(contours_field[0])
    new_x = x - 100  
    new_y = y - 270  
    new_w = w + 520  
    new_h = h + 160  
    
    new_x2 = x - 100  
    new_y2 = y - 260  
    new_w2 = w + 420  
    new_h2 = h + 150  
    
    cv2.rectangle(contour_frame, (new_x + 10, new_y - 0), (new_x + new_w, new_y + new_h), (255, 0, 0), 2)
    cv2.rectangle(contour_frame, (new_x2 + 50, new_y2 - 0), (new_x2 + new_w2, new_y2 + new_h2), (0, 0, 255), 2)
        
    pin_contours = []
    for contour in contours_pins:
        area = cv2.contourArea(contour)
        if 0 < area < 1600:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / float(w)
            if 0.8 < aspect_ratio < 5:
                pin_contours.append(contour)
                
                # Add a small red dot at the center of each pin contour
                center_x, center_y = x + w // 2, y + h // 2
    
    # Filter pins inside the adjusted rectangle and exclude areas near the edges
    edge_buffer = 7  # Buffer distance from the edge
    filtered_pins = [
        contour for contour in pin_contours 
        if new_x + edge_buffer + 30 <= cv2.boundingRect(contour)[0] <= new_x + new_w - edge_buffer - 10
        and new_y + edge_buffer <= cv2.boundingRect(contour)[1] <= new_y + new_h - edge_buffer 
    ]
    num_pins = len(filtered_pins)

    cv2.putText(contour_frame, f'Pins: {num_pins}', (14,  465), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    
    for contour in filtered_pins:
        x, y, w, h = cv2.boundingRect(contour)
        center_x, center_y = x + w // 2, y + h // 2
        cv2.circle(contour_frame, (center_x, center_y), 2, (0, 0, 255), -1)
        pin_center.append((center_x, center_y))
        smallest_pincenter = min(pin_center, key=lambda point: point[0]) 
           
    # Draw the points_list red dots with size 15 within the blue rectangle
    for point in points_list:
        if new_x + 10 <= point[0] <= new_x + new_w and new_y <= point[1] <= new_y + new_h:
            points_select.append((point[0], point[1]))
    
    # Find the point with the smallest x value in points_select and draw it
    if points_select:
        smallest_x_point = min(points_select, key=lambda point: point[0])
        cv2.circle(contour_frame, smallest_x_point, 15, (0, 0, 255), -1)
        
    # Calculate distance between the first two points
    if len(pin_center) != 0:
        if len(points_list) >= 2:
            pt1 = smallest_pincenter
            pt2 = smallest_x_point
            distance = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            cv2.line(contour_frame, pt1, pt2, (255, 0, 0), 2)
            cv2.putText(contour_frame, f'Distance: {distance:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # # Draw bounding boxes for all detected pin contours
    # for contour in pin_contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(contour_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Green for all detected pins

    # # Draw bounding boxes for filtered pins
    # for contour in filtered_pins:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(contour_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for filtered pins

    return contour_frame, num_pins, cropped_frame.shape

# Load the new image
new_image_path = 'pin/dataset_carc/F/in_calibrateline/image38.jpg' # E 30  # C 20
new_image = load_image(new_image_path)
if new_image is None:
    raise ValueError("Error: Could not open the newly uploaded image.")

# Process the new image to detect pins
adjusted_contour_frame, adjusted_num_pins, shape = refine_pin_detection_adjusted(new_image)
# Display the result
cv2.namedWindow('Refined Contours', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Refined Contours', 1300, 600)  # Resize window to 1300x600 pixels
cv2.imshow('Refined Contours', adjusted_contour_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
adjusted_output_path = '/mnt/data/AdjustedRefinedContourPin_image_with_boxes.png'
cv2.imwrite(adjusted_output_path, adjusted_contour_frame)

adjusted_output_path, adjusted_num_pins
