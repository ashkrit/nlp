from openai import OpenAI
import os
        
DATABRICKS_TOKEN = os.getenv('db_api')
        
client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url="https://dbc-a5e1d364-4fc3.cloud.databricks.com/serving-endpoints"
)
        
chat_completion = client.chat.completions.create(
  messages=[
  {
    "role": "system",
    "content": "You are an AI assistant"
  },
  {
    "role": "user",
    "content": "Tell me about Large Language Models"
  }
  ],
  model="databricks-dbrx-instruct",
  max_tokens=256
)
        
print(chat_completion.choices[0].message.content)