import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"


img = cv2.imread(r"./MeterPictures/pic2.jpg")
grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
threshold = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 655, 1)

cropped = threshold[400:600,200:1200]
cropped = cv2.medianBlur(cropped, 7)
cropped = cv2.bitwise_not(cropped)


text = pytesseract.image_to_string(cropped, config="outputbase digits")
file = open("recognized.csv","a")
file.write(text)
file.close()