
# Python program to explain
# mask inversion on a b/w image.

# importing cv2 and numpy library
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
# Reading an image
img = cv2.imread('/content/istockphoto-143918363-612x612.jpg')

# mask=cv2.imread('mask.jpeg')
mask = cv2.imread('mask.jpeg' , cv2.IMREAD_GRAYSCALE)
# print()
# The kernel to be used for dilation purpose
kernel = np.ones((5, 5), np.uint8)

# converting the image to HSV format
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# defining the lower and upper values of HSV,
# this will detect yellow colour
Lower_hsv = np.array([20, 70, 100])
Upper_hsv = np.array([30, 255, 255])
# Lower_hsv = np.array([0, 0, 0])
# Upper_hsv = np.array([255, 255, 0])

# creating the mask by eroding,morphing,
# dilating process
Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
print(Mask.shape)
Mask = cv2.erode(Mask, kernel, iterations=1)
Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
Mask = cv2.dilate(Mask, kernel, iterations=1)

# Inverting the mask by
# performing bitwise-not operation
# Mask = cv2.bitwise_not(Mask)
cv2_imshow(img)
# Displaying the image
cv2_imshow(Mask)
cv2.imwrite('apple_mask.png', Mask)
# waits for user to press any key
# (this is necessary to avoid Python
# kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
mask.shape,Mask.shape

import numpy as np
import cv2

# Open the image.
# img = cv2.imread('cat_damaged.png')

# Load the mask.
# mask = cv2.imread('cat_mask.png', 0)

# Inpaint.
dst = cv2.inpaint(img, mask, 10, cv2.INPAINT_NS)
cv2_imshow(dst)
dst2 = cv2.inpaint(img, mask, 100, cv2.INPAINT_NS)
cv2_imshow(dst2)
# Write the output.
# cv2.imwrite('cat_inpainted.png', dst)



# Python program to explain
# mask inversion on a b/w image.

# importing cv2 and numpy library
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
# Reading an image
img = cv2.imread('apples.jpg')

# The kernel to be used for dilation purpose
kernel = np.ones((5, 5), np.uint8)

# converting the image to HSV format
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# defining the lower and upper values of HSV,
# this will detect yellow colour
# Lower_hsv = np.array([20, 70, 100])
# Upper_hsv = np.array([30, 255, 255])
Lower_hsv = np.array([0, 0, 0])
Upper_hsv = np.array([10, 10, 10])

# creating the mask by eroding,morphing,
# dilating process
Mask = cv2.inRange(img, Lower_hsv, Upper_hsv)
cv2_imshow(Mask)
# Mask = cv2.erode(Mask, kernel, iterations=1)
# Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
# Mask = cv2.dilate(Mask, kernel, iterations=1)

# Inverting the mask by
# performing bitwise-not operation
# Mask = cv2.bitwise_not(Mask)
# cv2_imshow(img)
# Displaying the image
# cv2_imshow(Mask)

# waits for user to press any key
# (this is necessary to avoid Python
# kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()



import numpy as np
import cv2

# Open the image.
# img = cv2.imread('cat_damaged.png')

# Load the mask.
# mask = cv2.imread('cat_mask.png', 0)
cv2_imshow(img)
# Inpaint.
dst = cv2.inpaint(img, Mask, 3, cv2.INPAINT_NS)
cv2_imshow(dst)

dst2 = cv2.inpaint(img,Mask,3,cv2.INPAINT_TELEA)
cv2_imshow(dst2)
# Write the output.
# cv2.imwrite('cat_inpainted.png', dst)













