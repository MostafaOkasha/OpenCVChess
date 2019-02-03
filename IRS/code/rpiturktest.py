import cv2
from IPython import embed
import PIL
import numpy as np
from pprint import pprint
import sys

SQUARE_SIZE = 150
BOARD_SIZE = SQUARE_SIZE * 8
# imgpath = "/Users/robing/Desktop/test.jpg"
# imgpath = input("Enter the path to the image: ")
imgpath = sys.argv[1]
img = cv2.imread(imgpath)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Threshold for determining white color
white_sensitivity = 30
lower_white = np.array([0, 0, 150])
upper_white = np.array([128, white_sensitivity, 255])

white_threshold = 18000

# Threshold for determining green color
green_sensitivity = 40
lower_green = np.array([100 - green_sensitivity, 25, 25])
upper_green = np.array([100 + green_sensitivity, 255, 255])
green_threshold = 20000

pieces_range = list(range(0, 16)) + list(range(48, 64))
no_pieces = range(16, 48)
all_pieces = range(64)

# embed()
location_matrix = [[None] * 8,
                   [None] * 8,
                   [None] * 8,
                   [None] * 8,
                   [None] * 8,
                   [None] * 8,
                   [None] * 8,
                   [None] * 8]

for i in all_pieces:
    y = i % 8 * SQUARE_SIZE
    x = i // 8 * SQUARE_SIZE

    crop = hsv[x: x + SQUARE_SIZE, y: y + SQUARE_SIZE]

    white_mask = cv2.inRange(crop, lower_white, upper_white)
    green_mask = cv2.inRange(crop, lower_green, upper_green)

    white_values = (white_mask == 255).sum()
    green_values = (green_mask == 255).sum()

    piece_exists = (white_values < white_threshold) and (green_values < green_threshold)
    print("Square=", (i // 8, i % 8), "White values=", white_values, ", Green values=", green_values, "Piece is there=", piece_exists)

    # cv2.imshow("image", img[x: x + SQUARE_SIZE, y: y + SQUARE_SIZE])
    # cv2.waitKey(0)

    location_matrix[(i // 8)][i % 8] = piece_exists

print()
pprint(location_matrix)
