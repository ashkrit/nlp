from transformers import pipeline
from transformers import AutoTokenizer
from transformers import BertConfig, BertModel

import sys


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

config = BertConfig()
model = BertModel(config)

print(config)

model = BertModel.from_pretrained("bert-base-cased")

print(model)

## model.save_pretrained("/Users/ashkrit/_model")


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)

decoded_string = tokenizer.decode(ids)
print(decoded_string)
