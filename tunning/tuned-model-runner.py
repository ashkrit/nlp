import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification,pipeline
import sys

local_model_path = sys.argv[1]
print("Loading Model " , local_model_path)
retained_model = AutoModelForSequenceClassification.from_pretrained(local_model_path,local_files_only=True)

defaultclassifier  = pipeline("fill-mask" ,"bert-base-uncased")
trainedclassifier= pipeline("fill-mask",retained_model.name_or_path)

print("What do you want to fill>")
for line in sys.stdin:
    if 'exit' == line.rstrip():
        break
    result1 = defaultclassifier(line.rstrip())
    result2 = trainedclassifier(line.rstrip())
    
    for r in result1:
        print(r)

    print("***")

    for r in result2:
        print(r)    

    print("What do you want to fill>")


