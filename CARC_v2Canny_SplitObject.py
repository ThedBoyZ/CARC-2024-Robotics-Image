import cv2
import numpy as np

# Load a Video
cap = cv2.VideoCapture('/mnt/data/pin/Pin_bowling.mp4')

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
    
    # Pre-process the Frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge Detection for Field Boundaries
    edges = cv2.Canny(blur, 30, 150)

    # Find Contours for Field Boundaries
    contours_field, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Color Filtering for Pins
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Find Contours for Pins
    contours_pins, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw Contours on the Frame
    contour_frame = frame.copy()  # Start with the original frame
    cv2.drawContours(contour_frame, contours_field, -1, (0, 255, 0), 2)  # Green for field
    cv2.drawContours(contour_frame, contours_pins, -1, (0, 0, 255), 2)   # Red for pins

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
