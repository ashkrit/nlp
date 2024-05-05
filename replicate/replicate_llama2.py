import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Replicate
from io import BytesIO

import requests
from PIL import Image

key = os.getenv('replit_key')
os.environ["REPLICATE_API_TOKEN"] = key

text2image = Replicate(
    model="stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
    model_kwargs={"image_dimensions": "512x512"},
)



## read text from console 

while True:
    text = input("Enter text: ")
    if text == "exit":
        break
    image_output = text2image(text)
    print(f"Path {image_output}")
    response = requests.get(image_output)
    img = Image.open(BytesIO(response.content))
    img.show()    

