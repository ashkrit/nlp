from transformers import pipeline
import pandas as pd
import sys

classifier = pipeline("text-classification")

print("How are you feeling>")

for line in sys.stdin:
    if 'exit' == line.rstrip():
        break
    result = classifier(line.rstrip())
    print(result)
    print("How are you feeling>")