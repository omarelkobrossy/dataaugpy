import cv2
import numpy as np

# Load the image
image = cv2.imread('image.jpg')

# Get the dimensions of the image
height, width, _ = image.shape

# Define the bulge parameters (you can adjust these values)
bulge_strength = 1  # Adjust the strength of the bulge effect
center_x = width // 4  # X-coordinate of the center of the bulge
center_y = height // 4  # Y-coordinate of the center of the bulge

# Create a distortion map
distorted_map_x = np.zeros((height, width), dtype=np.float32)
distorted_map_y = np.zeros((height, width), dtype=np.float32)

for y in range(height):
    for x in range(width):
        # Calculate the distance from the center
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx ** 2 + dy ** 2)
        
        # Apply the bulge distortion
        if distance < center_x:
            scale = 1 - (distance / center_x) ** (1 + bulge_strength)
            new_x = x - dx * scale
            new_y = y - dy * scale
            if 0 <= new_x < width and 0 <= new_y < height:
                distorted_map_x[y, x] = new_x
                distorted_map_y[y, x] = new_y
            else:
                distorted_map_x[y, x] = x
                distorted_map_y[y, x] = y
        else:
            distorted_map_x[y, x] = x
            distorted_map_y[y, x] = y

# Apply the bulge warp using cv2.remap
bulged_image = cv2.remap(image, distorted_map_x, distorted_map_y, interpolation=cv2.INTER_LINEAR)

# Display the bulged image
cv2.imshow('Bulged Image', bulged_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the bulged image (optional)
cv2.imwrite('bulged_image.jpg', bulged_image)
