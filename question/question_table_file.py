from transformers import pipeline


import sys
import pandas as pd

pipe = pipeline("table-question-answering",model="google/tapas-medium-finetuned-wtq")

table = pd.read_csv("/Users/ashkrit/_code/nlp/question/purchase_data.csv")
table = table.astype(str)

print("Ask Question > ")
for input in sys.stdin:
    answers = pipe(table=table, query=input.strip())
    print(answers)
    for r in answers:
        print(r)
    print("Ask Question > ")
