import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np
from datasets import list_datasets, load_dataset
from transformers import AutoTokenizer
from transformers import AutoModel
import torch

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k: v.to(device) for k, v in batch.items()
              if k in tokenizer.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


emotions = load_dataset("emotion")
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

emotions_encoded.set_format(
    "torch", columns=["input_ids", "attention_mask", "label"])
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)


X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])


lr_clf = LogisticRegression(max_iter=3000)
classifier = lr_clf.fit(X_train, y_train)

model_location = f"/Users/ashkrit/_tmp/model/{model_ckpt}-feature-model.pkl"
with open(model_location, "wb") as f:
    pickle.dump(classifier, f)

print(f"Written at {model_location}")


