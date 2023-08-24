import sys
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print("How are you feeling>")
for line in sys.stdin:
    if 'exit' == line.rstrip():
        break
    result = classifier(line.rstrip())
    print(result)
    print("How are you feeling>")
