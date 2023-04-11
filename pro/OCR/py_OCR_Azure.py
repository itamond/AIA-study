from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image,  ImageDraw, ImageFont
import time
import openai
openai.api_key = "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"

subscription_key = "2f32a3b24bce4f67a376a19ad6941bed"
endpoint = "https://baehwanhe.cognitiveservices.azure.com/"

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# 이미지 가져오기
read_image_path = "https://mblogthumb-phinf.pstatic.net/MjAxOTA1MDFfNTkg/MDAxNTU2NzA1Mzg5NjI3.A1n3hHyGQutPjx8zjRTwW0QufjRaRof3OWMKEG1iTkYg.UTVRlKpp5H1KFUEfyQcQjb8IduG6Y8WyVS0QJ8Hi03Qg.JPEG.seel48/SE-28a7c1cb-5448-4c45-9a66-cb362af4ea87.jpg?type=w800"

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
img = Image.new('RGB', (width, height), bg_color)
draw = ImageDraw.Draw(img)
draw.text((10, 10), assistant_content, fill=text_color, font=font)
img.save('D:/number/cal_img/gpt_response_Azure.jpg')