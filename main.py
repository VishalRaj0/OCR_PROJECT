import cv2
import pytesseract
import numpy as np
# from pytesseract import Output
import matplotlib.pyplot as plt
from PIL import Image
from resize import image_resize


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


img = cv2.imread("images/i1.jpg")
#img = cv2.resize(img, (600, 360))
img = image_resize(img, 800, 800)


#1. GRAYSCALE
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 5. Erosion
kernel = np.ones((3,3), np.uint8)
# Erosion
erosion = cv2.erode(gray, kernel, iterations=1)
# Opening (erosion followed by dilation)
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
# Closing (dilation followed by erosion)
closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

#2. Deskewing (correct skew)
def deskew(image):
   gray = cv2.bitwise_not(image)  # Invert the grayscale image
   thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
   coords = np.column_stack(np.where(thresh > 0))
   angle = cv2.minAreaRect(coords)[-1]

   # Adjust angle
   if angle < -45:
       angle = -(90 + angle)
   else:
       angle = -angle


   (h, w) = image.shape[:2]
   center = (w // 2, h // 2)
   M = cv2.getRotationMatrix2D(center, angle, 1.0)
   return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Call deskew
deskewed = deskew(gray)


#3. on-local means denoising (remove noise) (added)
denoised = cv2.fastNlMeansDenoising(deskewed, None, 30, 7, 21)


#6. Canny Edge Detection
edges = cv2.Canny(deskewed, 100, 200)


# 4. Thresholding
#a) ADAPTIVE MEAN THRESHOLDING
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 4)
plt.subplot(2, 2, 1)
plt.title("ADAPTIVE MEAN THRESHOLDING")
plt.imshow(thresh)


#b) OTSU+BINARY INVERSE THRESHOLDING
ret, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
plt.subplot(2, 2, 2)
plt.title("OTSU+BINARY INVERSE")
plt.imshow(th1)


#c) OTSU THRESHOLDING
_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
plt.subplot(2, 2, 3)
plt.title("OTSU THRESHOLDING")
plt.imshow(th2)


#d) GAUSSIAN THRESHOLDING (blurring + Otsu thresholding)
blur = cv2.GaussianBlur(gray, (3,3), 0)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.subplot(2, 2, 4)
plt.title("GAUSSIAN THRESHOLDING")
plt.imshow(th3)
plt.show()


# #3. Thresholding
# #a) ADAPTIVE MEAN THRESHOLDING
# thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 4)
# plt.subplot(2, 2, 1)
# plt.title("ADAPTIVE MEAN THRESHOLDING")
# plt.imshow(thresh)
#
# #b) OTSU+BINARY INVERSE THRESHOLDING
# ret, th1 = cv2.threshold(denoised, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
# plt.subplot(2, 2, 2)
# plt.title("OTSU+BINARY INVERSE")
# plt.imshow(th1)
#
# #c) OTSU THRESHOLDING
# _, th2 = cv2.threshold(denoised, 0, 255, cv2.THRESH_OTSU)
# plt.subplot(2, 2, 3)
# plt.title("OTSU THRESHOLDING")
# plt.imshow(th2)
#
# #d) GAUSSIAN THRESHOLDING (blurring + Otsu thresholding)
# blur = cv2.GaussianBlur(denoised, (3,3), 0)
# _, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# plt.subplot(2, 2, 4)
# plt.title("GAUSSIAN THRESHOLDING")
# plt.imshow(th3)
# plt.show()


#7. DILATION
# a) Dilation on the original image:
kernel = np.ones((3,3), np.uint8)
dilation = cv2.dilate(img, kernel, iterations=1)
plt.subplot(1, 2, 1)
plt.title("DILATION ON ORIGINAL IMAGE")
plt.imshow(dilation)


# b) Dilation on the grayscale image
g_dilation = cv2.dilate(gray, kernel, iterations=1)
plt.subplot(1, 2, 2)
plt.title("DILATION ON GRAYSCALE")
plt.imshow(g_dilation)


plt.show()


rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))


img = img
if len(img.shape) == 3:
   hImg, wImg, _ = img.shape
else:
   hImg, wImg = img.shape


boxes = pytesseract.image_to_boxes(img)
with open("recognized.txt", "w") as fileobj:
   for b in boxes.splitlines():
       b = b.split(' ')
       x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
       cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (50, 50, 255), 1)
       cv2.putText(img, b[0], (x, hImg - y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


       fileobj.write(b[0])


cv2.imshow('Detected text', img)
cv2.waitKey(0)


# text = pytesseract.image_to_string(Image.open("images/img33.jpg"))
# text = pytesseract.image_to_string(denoised)
# text = pytesseract.image_to_string(gray)
# text = pytesseract.image_to_string(thresh)
# text = pytesseract.image_to_string(th1)
# text = pytesseract.image_to_string(th2)
text = pytesseract.image_to_string(deskewed)
# text = pytesseract.image_to_string(denoised)
print(text)