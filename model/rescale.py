import cv2
import numpy as np

# -----------------------------
# SETTINGS
# -----------------------------

INPUT_IMAGE = "real-images/five.jpg"
OUTPUT_MASK = "real-images/mask.png"

# Target size
SIZE = (50, 50)

# Example: black color mask in HSV
lower = np.array([0, 0, 0])      # lower HSV bound
upper = np.array([180, 255, 55])   # upper HSV bound

# -----------------------------
# PROCESSING
# -----------------------------

# 1. Load image
img = cv2.imread(INPUT_IMAGE)

# 2. Resize to 50×50 pixels
img_small = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)

# 3. Convert to HSV (better for color masking)
hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

# 4. Create mask
mask = cv2.inRange(hsv, lower, upper)

# 5. Save mask
cv2.imwrite(OUTPUT_MASK, mask)

print("Done! Mask saved to:", OUTPUT_MASK)
