import numpy as np
import cv2

def load_image(path_img):
    return cv2.imread(path_img)

def zoom(img, zoom_factor=1.5):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

def refine_pin_detection_adjusted(img):
    frame_width, frame_height = img.shape[1], img.shape[0]
    zoom_factor = 3
    zoomed_frame = zoom(img, zoom_factor)

    zoomed_height, zoomed_width = zoomed_frame.shape[:2]
    crop_x = (zoomed_width - frame_width) // 2
    crop_y = ((zoomed_height - frame_height) // 2) - 200

    cropped_frame = zoomed_frame[crop_y:crop_y+frame_height, crop_x:crop_x+frame_width]
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    gray_inverted = cv2.bitwise_not(blur)
    _, binary = cv2.threshold(gray_inverted, 97, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(gray_inverted, 97, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)

    edges = cv2.Canny(binary, 50, 180)
    contours_pins, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_field, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_frame = cropped_frame.copy()
    cv2.drawContours(contour_frame, contours_pins, -1, (0, 255, 0), 2)
    cv2.drawContours(contour_frame, contours_field, -1, (0, 0, 255), 2)

    # Adjust the coordinates of the bounding box
    x, y, w, h = cv2.boundingRect(contours_field[0])
    new_x = x - 50  # Adjusted to properly fit the pins  ---> - 50 px
    new_y = y - 300  # Adjusted to properly fit the pins ---> -300 px
    new_w = w + 1620  # Adjusted width ----> + 1620 px
    new_h = h + 50  # Adjusted height  ----> +   50 px

    cv2.rectangle(contour_frame, (new_x+320, new_y-280), (new_x + new_w , new_y + new_h), (255, 0, 0), 2)
    
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
                cv2.circle(contour_frame, (center_x, center_y), 5, (0, 0, 255), -1)

    # Filter pins inside the adjusted rectangle and exclude areas near the edges
    edge_buffer = 8  # Buffer distance from the edge
    filtered_pins = [
        contour for contour in pin_contours 
        if new_x + edge_buffer + 320 <= cv2.boundingRect(contour)[0] <= new_x + new_w - edge_buffer
        and new_y - edge_buffer - 280 <= cv2.boundingRect(contour)[1] <= new_y + new_h - edge_buffer
    ]
    num_pins = len(filtered_pins)

    cv2.putText(contour_frame, f'Pins: {num_pins}', (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    return contour_frame, num_pins

# Load the newly uploaded image
new_image_path = 'pin/Pindetect_Test.jpg'
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
