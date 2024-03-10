import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Replicate
from io import BytesIO

import requests
from PIL import Image

key = os.getenv('replit_key')

os.environ["REPLICATE_API_TOKEN"] = key



llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
)
prompt = """
User: Answer the following yes/no question by reasoning step by step. Can a dog drive a car?
Assistant:
"""
reply = llm(prompt)

print(reply)


text2image = Replicate(
    model="stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
    model_kwargs={"image_dimensions": "512x512"},
)


image_output = text2image("A cat riding a motorcycle by Picasso")
print(image_output)

response = requests.get(image_output)
img = Image.open(BytesIO(response.content))

img