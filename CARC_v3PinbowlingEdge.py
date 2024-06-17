import cv2
import numpy as np

def zoom(img, zoom_factor=1.5):
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

# Load a Video
cap = cv2.VideoCapture('pin/Pin_bowling.mp4')
# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the width and height of the frames in the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi files
out = cv2.VideoWriter('/mnt/data/output/ContourPin.mp4', fourcc, 20.0, (frame_width, frame_height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    print(f"Processing frame {frame_count}")

    # Zoom into the frame
    zoom_factor = 1.5
    zoomed_frame = zoom(frame, zoom_factor)

    # Calculate the coordinates for cropping the center of the zoomed image
    zoomed_height, zoomed_width = zoomed_frame.shape[:2]
    crop_x = (zoomed_width - frame_width) // 2
    crop_y = ((zoomed_height - frame_height) // 2) - 100  # Move center of crop by -100 in crop_y

    # Crop the zoomed frame to the original frame size centered
    cropped_frame = zoomed_frame[crop_y:crop_y+frame_height, crop_x:crop_x+frame_width]

    # Pre-process the Frame
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Invert the grayscale image
    gray_inverted = cv2.bitwise_not(blur)
    
    # Create a binary thresholded image
    _, binary = cv2.threshold(gray_inverted, 100, 255, cv2.THRESH_BINARY)
      
    # Edge Detection for Field Boundaries
    edges = cv2.Canny(binary, 40, 150)

    # Find Contours for Field Boundaries
    contours_field, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find Contours for Pins
    contours_pins, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw Contours on the Frame
    contour_frame = cropped_frame.copy()
    cv2.drawContours(contour_frame, contours_field, -1, (0, 255, 0), 2) # Green for field
    cv2.drawContours(contour_frame, contours_pins, -1, (0, 0, 255), 2) # Red for pins
    
    # Filter the contours to identify standing pins
    pin_contours = []
    for contour in contours_pins:
        area = cv2.contourArea(contour)
        # Filter based on area and aspect ratio to detect pin heads
        if 50 < area < 500:  # This threshold might need to be adjusted
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / float(w)
            if 1.5 < aspect_ratio < 5:
                pin_contours.append(contour)
    
    # Count the number of standing pins
    num_pins = len(pin_contours)
    
    cv2.putText(contour_frame, f'Pins: {num_pins}', (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Optionally display the frame
    cv2.imshow('Contours', contour_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Write the frame with contours
    out.write(contour_frame)

# Release everything if the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete. Video saved to '/mnt/data/output/ContourPin.mp4'")
