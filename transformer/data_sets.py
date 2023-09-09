from huggingface_hub import list_datasets
from huggingface_hub import list_models

for ds in list_datasets():
    print(ds)

print("******")

for ds in list_models():
    print(ds)