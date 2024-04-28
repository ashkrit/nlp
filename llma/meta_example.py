from meta_ai_api import MetaAI
from requests_html import HTMLSession

ai = MetaAI()

t = ai.get_access_token()
print(t)
response = ai.prompt(message="When ios next match in IPL")
print(response)