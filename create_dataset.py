import os
import numpy as np
import cv2

# Create folders if not exist
os.makedirs("dataset/healthy", exist_ok=True)
os.makedirs("dataset/diseased", exist_ok=True)

# Image size
img_size = 128

# Generate Healthy Images (Green leaves)
for i in range(30):
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img[:] = (0, np.random.randint(150,255), 0)  # Green color
    cv2.imwrite(f"dataset/healthy/healthy_{i}.jpg", img)

# Generate Diseased Images (Brown spots)
for i in range(30):
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img[:] = (0, np.random.randint(150,255), 0)
    
    # Add brown spot
    cv2.circle(img, (64,64), 30, (19,69,139), -1)
    
    cv2.imwrite(f"dataset/diseased/diseased_{i}.jpg", img)

print("Dataset Created Successfully!")
