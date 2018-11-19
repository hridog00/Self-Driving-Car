import matplotlib.pyplot as plt
import matplotlib.image as impimg
import numpy as np
import cv2
import math
from PIL import Image

def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    theta = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            theta = theta+math.atan2((y2-y1),(x2-x1))
    return img, theta


imagePath = 'image/curva4.jpg'
im = Image.open(imagePath)
width, height = im.size
region_of_interest_vertices = [
    (0, height),
    (0,height / 2), (width,height / 2),
    (width, height),
]

image = impimg.imread(imagePath)

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)


cropped = region_of_interest(image,np.array([region_of_interest_vertices],np.int32),)




gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cannyed_image = cv2.Canny(gray_image, 100, 200)

cropped_image = region_of_interest(cannyed_image,np.array([region_of_interest_vertices],np.int32),)
plt.figure()
plt.imshow(cropped_image)
plt.show()
lines = cv2.HoughLinesP(
    cropped_image,
    rho=6,
    theta=np.pi / 60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25
)
left_line_x = []
left_line_y = []
right_line_x = []
right_line_y = []
for line in lines:
    for x1, y1, x2, y2 in line:
        slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.

        if slope <= 0: # <-- If the slope is negative, left group.
            left_line_x.extend([x1, x2])
            left_line_y.extend([y1, y2])
        else: # <-- Otherwise, right group.
            right_line_x.extend([x1, x2])
            right_line_y.extend([y1, y2])
min_y = int(image.shape[0] * (3 / 5)) # <-- Just below the horizon
max_y = image.shape[0] # <-- The bottom of the image
poly_left = np.poly1d(np.polyfit(
    left_line_y,
    left_line_x,
    deg=1
))
left_x_start = int(poly_left(max_y))
left_x_end = int(poly_left(min_y))
poly_right = np.poly1d(np.polyfit(
    right_line_y,
    right_line_x,
    deg=1
))
right_x_start = int(poly_right(max_y))
right_x_end = int(poly_right(min_y))
line_image, theta = draw_lines(
    image,
    [[
        [left_x_start, max_y, left_x_end, min_y],
        [right_x_start, max_y, right_x_end, min_y],
    ]],
    thickness=5,
)

threshold=6
if(theta>threshold):

    print("left")
if(theta<-threshold):

    print("right")
if(abs(theta)<threshold):

    print( "straight")
plt.figure()
plt.imshow(line_image)
plt.show()