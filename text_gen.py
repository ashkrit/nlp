from transformers import pipeline
import sys


## generator = pipeline("text-generation")
generator = pipeline("text-generation", model="distilgpt2")

print("What do you want to say>")
for line in sys.stdin:
    result = generator(line.strip() , max_length=30, num_return_sequences=2)   
    for r in result: 
        print(r)
    print("What do you want to say>")
