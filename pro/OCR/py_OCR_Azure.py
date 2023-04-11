from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from array import array
from PIL import Image,  ImageDraw, ImageFont
import sys
import time
import os
import openai
openai.api_key = "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"

subscription_key = "2f32a3b24bce4f67a376a19ad6941bed"
endpoint = "https://baehwanhe.cognitiveservices.azure.com/"

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# 이미지 가져오기
read_image_path = "https://image.yes24.com/goods/89019624/XL"

# API 불러오기
read_response = computervision_client.read(read_image_path,  raw=True)


read_operation_location = read_response.headers["Operation-Location"]
operation_id = read_operation_location.split("/")[-1]

while True:
    read_result = computervision_client.get_read_result(operation_id)
    if read_result.status not in ['notStarted', 'running']:
        break
    time.sleep(1)
out_ocr=str()
if read_result.status == OperationStatusCodes.succeeded:
    for text_result in read_result.analyze_result.read_results:
        for line in text_result.lines:
            out_ocr+=line.text+'\n'

messages = []

print(out_ocr)

while True:
    user_content = out_ocr
    messages.append({"role": "user", "content": f"{user_content}"})

    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    assistant_content = completion.choices[0].message["content"].strip()

    messages.append({"role": "assistant", "content": f"{assistant_content}"})

    print(f"GPT : {assistant_content}")
    break


width, height = 1800, 1800
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
img.save('D:/number/cal_img/gpt_response_Azure.jpg')