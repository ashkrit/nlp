from langchain.document_loaders import PyPDFLoader
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

doc_path = "/Users/ashkrit/Downloads/188f1b7f-ff6d-4f62-a337-40a006076345.pdf"
logging.info("Loading %s Document", doc_path)

start_time = time.time()
chunks = PyPDFLoader(doc_path).load()
load_time = time.time() - start_time

logging.info("Took %s for loading", load_time)

#print(chunks[:5])

data = chunks[:10]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_splits = text_splitter.split_documents(data)

for chunk in all_splits:
    logging.info("Chunk Length : %s", len(chunk.page_content))
    logging.info("Chunk Text : %s", chunk.page_content)
    logging.info("*********")
