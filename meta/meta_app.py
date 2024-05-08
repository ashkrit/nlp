from meta_ai_api import MetaAI
import os

p = os.getenv("fb_password")
print(p)
ai = MetaAI(fb_email="ashkrit@gmail.com",fb_password=p)
response = ai.prompt(message="create image of happy people")
print(response)