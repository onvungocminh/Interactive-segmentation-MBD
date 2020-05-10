

import cv2
import numpy as np
from queue import PriorityQueue
import copy
import matplotlib.pyplot as plt



def waterflow_distance(img, fg_markers):
    # Initiate values
    # Put markers in the priority queue

    state = np.zeros((h, w))
    dis_map = np.zeros_like(img)
    Q = PriorityQueue()
    min_image = np.zeros_like(img)
    max_image = np.zeros_like(img)

    for i in range(0, h):
        for j in range(0, w):
            min_image[i, j] = img[i, j]
            max_image[i, j] = img[i, j]
            if [i,j] in fg_markers:
                state[i, j] = 1
                dis_map[i, j] = (0, 0, 0)
                Q.put((sum(dis_map[i, j]), [i, j]))


            else:
                state[i, j] = 0
                dis_map[i, j] = (255, 255, 255)

    # Propagation
    neighbor = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    while not Q.empty():
        p = Q.get()[1]
        if state[p[0], p[1]] == 2:
            continue
        state[p[0], p[1]] = 2

        for n1 in range(len(neighbor)):
            x = p[0] + neighbor[n1][0]
            y = p[1] + neighbor[n1][1]
            if x >= 0 and x < h and y >= 0 and y < w:

                if state[x, y] == 1 and sum(dis_map[x, y]) > sum(dis_map[p[0], p[1]]):
                    min_image[x, y] = min_image[p[0], p[1]]
                    max_image[x, y] = max_image[p[0], p[1]]
                    first = np.minimum(min_image[x, y], img[x, y])
                    second = np.maximum(max_image[x, y], img[x, y])
                    if sum(dis_map[x, y]) > sum(max_image[x, y] - min_image[x, y]):
                        dis_map[x, y] = second - first
                        Q.put((sum(dis_map[x, y]), [x, y]))

                elif state[x, y] == 0:
                    min_image[x, y] = min_image[p[0], p[1]]
                    max_image[x, y] = max_image[p[0], p[1]]
                    first = np.minimum(min_image[x, y], img[x, y])
                    second = np.maximum(max_image[x, y], img[x, y])
                    dis_map[x, y] = second - first
                    Q.put((sum(dis_map[x, y]), [x, y]))
                    state[x, y] = 1
                else:
                    continue

    dis_map = np.array(dis_map, dtype="uint8")  # convert to uint8
    dis_gray = cv2.cvtColor(dis_map, cv2.COLOR_BGR2GRAY)
    return dis_gray

# load image

src_name = 'Imgs/227092.jpg'
label_name = 'Imgs/227092-anno.png'


img = cv2.imread(src_name)
label_gray = cv2.cvtColor(cv2.imread(label_name), cv2.COLOR_BGR2GRAY)
h,w = img.shape[:2]


# markers

bg_markers = []
fg_markers = []

for i in range(0, h):
    for j in range(0, w):
        if (label_gray[i][j] > 10 and label_gray[i][j] < 100):
            bg_markers.append([i, j])

        if (label_gray[i][j] > 100):
            fg_markers.append([i, j])

dis_map_fg = waterflow_distance(img, fg_markers)
dis_map_bg = waterflow_distance(img, bg_markers)

segment = np.zeros((h,w))
segment[dis_map_fg<= dis_map_bg] = 255

#  Post processing

label_gray[label_gray > 100] = 255

segment = np.array(segment, dtype="uint8")  # convert to uint8
# Find largest contour in intermediate image so that it contains the markers
cnts, _ = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
con = []
for i in range(len(cnts)):
    biggest = np.zeros(segment.shape, np.uint8)
    cv2.drawContours(biggest, [cnts[i]], -1, 255, cv2.FILLED)
    biggest = cv2.bitwise_and(label_gray, biggest)
    if (np.sum(biggest) > 0):
        con.append(cnts[i])

print(len(con))

biggest = np.zeros(segment.shape, np.uint8)

if (len(con) != 0):
    cnt = max(con, key=cv2.contourArea)

    # Output
    cv2.drawContours(biggest, [cnt], -1, 255, cv2.FILLED)

print(np.max(biggest))
biggest = np.array(biggest, dtype="uint8")  # convert to uint8

f = plt.figure(1)
plt.imshow(dis_map_fg)

g = plt.figure(2)
plt.imshow(dis_map_bg)

k = plt.figure(3)
plt.imshow(segment)

l = plt.figure(4)
plt.imshow(biggest)

plt.show()

