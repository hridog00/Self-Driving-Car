import matplotlib.pyplot as plt
import matplotlib.image as impimg
import numpy as np
import cv2
import math
from PIL import Image

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
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

#imagePath = 'test_images/curvadere.jpg'

#imagePath = 'test_images/curvaizquierda.jpg'
#imagePath = 'test_images/curvaderecha.jpg'
#imagePath = 'test_images/recta.jpg'
#imagePath = 'test_images/semibuena.jpg'
imagePath = 'test_images/picture3.jpg'




im = Image.open(imagePath)
width, height = im.size
region_of_interest_vertices = [
    (0, height),
    (0, height ),(width, height),
    (width, height),
]

i = impimg.imread(imagePath)
image = i[2:384, 0:5400]

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)


plt.figure()
plt.imshow(image)

image = increase_brightness(image, value=80)
plt.figure()
plt.imshow(image)
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.figure()
plt.imshow(gray_image)


cannyed_image = cv2.Canny(gray_image, 100, 200)
plt.figure()
plt.imshow(cannyed_image)
cropped_image = region_of_interest(cannyed_image,np.array([region_of_interest_vertices],np.int32),)
plt.figure()
plt.imshow(cannyed_image)
size = cropped_image.shape
#Cada 20 lineas, calcular distancia al 255 por la izquierda y por la derecha
#izda = distizda + izda, lo mismo con la dcha (range(10, 0, -1))
#ida/mediciones realizadas, lo mismo con la dcha
#tomar decision
cropped_image = cannyed_image
muestras = 0
dist_izda = 0
dist_dcha = 0
print(size[0])
for i in range(100,0, -10):

    izda = 0
    for x in (range(int(size[1]/2), 0, -1)):
        if cropped_image[i][x] == 255:
            break
        else:
            izda = izda +1
    dcha = 0
    for x in (range(int(size[1] / 2), size[1])):
        if cropped_image[i][x] == 255:
            break
        else:
            dcha = dcha + 1
    dist_izda = dist_izda + izda
    dist_dcha = dist_dcha + dcha
    muestras = muestras +1

dist_dcha = dist_dcha/muestras
dist_izda = dist_izda/muestras



print(cropped_image.shape)
print(dist_izda, dist_dcha)
if (abs(dist_izda - dist_dcha) >= 50):
    if (dist_izda > dist_dcha):
        print('left')
    else:
        print('right')
else:
    print('straight')

plt.show()

#recta: 472, 381
#curvadcha: 521, 511
#curvaizda:  593, 293
#semibuena: 435,426