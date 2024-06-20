import cv2

# Assuming contours_field and screen dimensions are defined
x, y, w, h = cv2.boundingRect(contours_field[0])
print("Original coordinates and dimensions:")
print(f"x: {x}, y: {y}, w: {w}, h: {h}")

# Adjustments as provided
new_x = x + 121
new_y = y + 200
new_w = w + 520  
new_h = h - 340  

new_x2 = x + 121  
new_y2 = y + 220 
new_w2 = w + 420  
new_h2 = h - 380  

# Dimensions of the screen or image
screen_width = 1920  # Example value
screen_height = 1080  # Example value

# Center of the screen
screen_center_x = screen_width // 2
screen_center_y = screen_height // 2

# Center of the original bounding box
box_center_x = x + w // 2
box_center_y = y + h // 2

# Calculate offset to center the bounding box on the screen
offset_x = screen_center_x - box_center_x
offset_y = screen_center_y - box_center_y

# Adjust the coordinates to center the bounding box
centered_x = x + offset_x
centered_y = y + offset_y

# Apply the same offset to the second box
centered_x2 = centered_x
centered_y2 = centered_y + 20  # Assuming a fixed vertical shift for the second box

# Print the adjusted coordinates
print("Centered coordinates and dimensions:")
print(f"Box 1 - x: {centered_x}, y: {centered_y}, w: {new_w}, h: {new_h}")
print(f"Box 2 - x: {centered_x2}, y: {centered_y2}, w: {new_w2}, h: {new_h2}")

# If you need to draw the rectangles on an image
# Assuming 'image' is your image array
image = cv2.imread('your_image_path_here')
cv2.rectangle(image, (centered_x, centered_y), (centered_x + new_w, centered_y + new_h), (255, 0, 0), 2)
cv2.rectangle(image, (centered_x2, centered_y2), (centered_x2 + new_w2, centered_y2 + new_h2), (0, 255, 0), 2)

# Show the image
cv2.imshow('Centered Rectangles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()