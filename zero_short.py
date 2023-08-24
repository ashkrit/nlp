from transformers import pipeline
import sys 

classifier = pipeline("zero-shot-classification")

print ("How are you feeling > ")
for line in sys.stdin:
    result = classifier(
        line.strip(),
        candidate_labels=["education", "politics", "business"],
    )
    print(result)
    print ("How are you feeling > ")

