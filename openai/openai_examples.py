from openai import OpenAI
import os
import pandas as pd
import tiktoken


def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

client = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))

base_path="~/Downloads/"
df = pd.read_csv(f"{base_path}/Amazon_Review/Reviews.csv" , index_col=0)
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()
df["combined"] = (
    "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)

print(df.head(5))

top_n = 1000
df = df.sort_values("Time").tail(10)  # first cut to first 2k entries, assuming less than half will be filtered out
df.drop("Time", axis=1, inplace=True)

df["embedding"] = df.combined.apply(lambda x: get_embedding(x))
df.to_csv(f"{base_path}/Amazon_Review/data/fine_food_reviews_with_embeddings_1k.csv")

