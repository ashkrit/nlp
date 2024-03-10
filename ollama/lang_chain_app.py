from langchain_community.llms import Ollama

llm = Ollama(model="llama2")

result = llm.invoke("Tell me a joke")

print(result)