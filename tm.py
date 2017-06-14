import cv2 #memanggil library opencv
import numpy as np #memanggil library numeric python
#size = 2
img_rgb = cv2.imread('gambar2.jpg')#gambar utama untuk diolah
img = cv2.imread('gambar2.jpg')#gambar utama untuk ditampilkan saja
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) #mengubah gambar ke dalam grayscale atau keabu abuan

template = cv2.imread('face2.jpg',1)#gambar wajah yang akan di cocokkan
tem_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) #megubah gambar wajah kedalam grayscale untuk mempermudah pengolahan
w, h = tem_gray.shape[::-1] #mencari nilai lebar dan panjang dari gambar wajah

#mini = cv2.resize(img_rgb, (img_rgb.shape[1] * size, img_rgb.shape[0] * size))
#img_gray = cv2.cvtColor(mini, cv2.COLOR_BGR2GRAY)
res = cv2.matchTemplate(img_gray,tem_gray,cv2.TM_CCOEFF_NORMED) #mencari matriks untuk mencocokan grayscale gambar utama dan grayscale gambar wajah 

threshold = 0.7 #sebagai pengatur kepekaan pencocokan gambar
loc = np.where( res >= threshold) #pencocokan gambar utama dan wajah

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (255,0,0), 2) #menampilkan marking berupa kotak pada wajah yang dianggap sama atau cocok

cv2.imshow('Result',img_rgb) #menampilkan gambar hasil
cv2.imshow('Target',template) #menampilkan gambar wajah
cv2.imshow('Image',img) #menampilkan gambar asli

cv2.waitKey(0)
cv2.destroyAllWindows()


#hasil 2 threshold 0.7
#hasil 3 threshold 0.6
#hasil 4 threshold 0.7
