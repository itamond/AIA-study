import cv2
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

path = 'C:/archive (3)/numbers/dd.png'
image = cv2.imread(path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# print(rgb_image)
text = pytesseract.image_to_string(rgb_image, lang='kor+eng')



print(text)