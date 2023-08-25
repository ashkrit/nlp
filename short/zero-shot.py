from transformers import pipeline
import sys

location="~/_model/facebook_bart-large-mnli"

classifier = pipeline("zero-shot-classification",model=location, local_files_only=True)


# classifier.save_pretrained(location)

print("Saved at " , location)


sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing','study','playing','rest']

print("How are you feeling >")

for text in sys.stdin:
    sequence_to_classify = text.strip()
    result = classifier(sequence_to_classify, candidate_labels)
    print(result)
    print("How are you feeling >")










