from transformers import pipeline 
import sys 

print("Speak your mind > ")
question_answerer = pipeline("question-answering")

for line in sys.stdin:

    result = question_answerer(question=line.strip(),context="My name is Sylvain and I work at Hugging Face in Brooklyn")
    print(result)
    print("Speak your mind > ")