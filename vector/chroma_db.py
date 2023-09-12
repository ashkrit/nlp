import chromadb
import io
import sys


chroma_client = chromadb.Client()

chroma_client = chromadb.PersistentClient(
    path="/Users/ashkrit/_tmp/vectordb/chroma2")
collection = chroma_client.get_or_create_collection(name="my_collection")

"""
file_name = "AMAZON_FASHION.json"
line_count = 0
lines = []
for line in io.open(f"/Users/ashkrit/_tmp/data/{file_name}"):
    lines.append(line.strip())
    line_count += 1
    if (line_count >= 10000):
        break


print("Lines loaded ", line_count)

ids = [f"{file_name}_{x}" for x in range(line_count)]
sources = [{"source": file_name} for x in range(line_count)]
documents = lines

print("Read to load")


chunk_size = 100
for index in range(0, line_count, chunk_size):
    end_chunk = index+chunk_size
    print(f"Loading from {index} to {end_chunk}")
    chunk_doc = documents[index:end_chunk]
    chunk_source = sources[index:end_chunk]
    chunk_ids = ids[index:end_chunk]
    collection.add(documents=chunk_doc, metadatas=chunk_source, ids=chunk_ids)

print("Data loaded")
"""

print("Ask Question>")
for input in sys.stdin:
    input = input.strip()
    results = collection.query(query_texts=[input], n_results=10)
    for r in results["documents"][0]:
        print(r)
        print()
    print("Ask Question>")
