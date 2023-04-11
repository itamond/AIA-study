import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import openai
from PIL import Image, ImageDraw, ImageFont
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


openai.api_key = "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"
path = 'D:/number/cal_img/question/cal_r.jpg'
image = cv2.imread(path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# print(rgb_image)
text = pytesseract.image_to_string(rgb_image, lang='kor+eng')

print(text)

messages = []

while True:
    user_content = text
    messages.append({"role": "user", "content": f"{user_content}"})

    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    assistant_content = completion.choices[0].message["content"].strip()

    messages.append({"role": "assistant", "content": f"{assistant_content}"})

    print(f"GPT : {assistant_content}")
    break

# Set up image parameters
width, height = 180, 180
bg_color = (255, 255, 255)
text_color = (0, 0, 0)
font_size = 60
font = ImageFont.truetype('malgun.ttf', font_size)

# Create an image with a white background
img = Image.new('RGB', (width, height), bg_color)

# Draw the GPT response onto the image
draw = ImageDraw.Draw(img)
draw.text((10, 10), assistant_content, fill=text_color, font=font)

# Save the image to disk
img.save('D:/number/cal_img/gpt_response.jpg')