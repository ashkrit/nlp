from transformers import pipeline
import pandas as pd

classifier = pipeline("text-classification")

reviews = pd.read_json(
    "/Users/ashkrit/_tmp/data/AMAZON_FASHION.json", lines=True)

print(reviews.columns)

reviews['comments'] = reviews['reviewText'].str.strip().replace('\n', ' ')
no_of_rows = 100

topReviews = reviews[['comments']].iloc[:no_of_rows]

pd.set_option('display.max_colwidth', None)

outputs = classifier(topReviews["comments"].to_list())
result = pd.DataFrame(outputs)

for r in range(0, len(result)):
    print("{} = [{}] -> {}".format(r, result.iat[r, 0], topReviews.iat[r, 0]))
