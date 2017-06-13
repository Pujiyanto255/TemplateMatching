import cv2
import numpy as np
size = 2
img_rgb = cv2.imread('gambar2.jpg')
img = cv2.imread('gambar2.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('face2.jpg',1)
tem_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
w, h = tem_gray.shape[::-1]

mini = cv2.resize(img_rgb, (img_rgb.shape[1] * size, img_rgb.shape[0] * size))
#img_gray = cv2.cvtColor(mini, cv2.COLOR_BGR2GRAY)
res = cv2.matchTemplate(img_gray,tem_gray,cv2.TM_CCOEFF_NORMED)
threshold = 0.7
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

cv2.imshow('Result',img_rgb)
cv2.imshow('Target',template)
cv2.imshow('Image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()


#hasil 2 threshold 0.7
#hasil 3 threshold 0.6
#hasil 4 threshold 0.7
