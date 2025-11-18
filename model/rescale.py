import cv2
import numpy as np

INPUT_IMAGE = "real-images/2.jpeg"
OUTPUT_MASK = "real-images/mask.png"

SIZE = (50, 50)

# HSV bounds
lower = np.array([0, 20, 50])      
upper = np.array([255, 255, 255])

img = cv2.imread(INPUT_IMAGE)

# 2. Resize to 50×50 pixels
#img_small = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)

# 3. Convert to HSV (better for color masking)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 4. Create mask
mask = cv2.inRange(hsv, lower, upper)

# 5. Save mask
cv2.imwrite(OUTPUT_MASK, mask)