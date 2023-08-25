from transformers import pipeline
from PIL import Image

pipe = pipeline("document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa")

question = "what labels are used  ?"
image = Image.open("~/Downloads/bill.png")

result = pipe(image=image, question=question)

print(result)