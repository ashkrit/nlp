from datasets import list_datasets,load_dataset
from transformers import AutoTokenizer
from transformers import AutoModel
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification

## Tokenizer 
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

print(f"Tokenizer {tokenizer}")

## Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device {device}")

num_labels = 6
id2label = {0:"sadness", 1:"joy", 2:"love", 3:"anger", 4:"fear", 5:"surprise"}
label2id = {val: key for key, val in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels,id2label=id2label, label2id=label2id).to(device)
print(f"Model {model}")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

emotions = load_dataset("emotion")
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
output_location="~/_tmp/model/finetuned-emotion"
training_args = TrainingArguments(output_dir=output_location,
                                  num_train_epochs=2,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  push_to_hub=False, ## for publishing 
                                  log_level="error")

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)

print(f"Training {trainer}")

tresult=trainer.train();

print(f"Training results {tresult}")

print(f"Saving to disk {output_location}")

trainer.save_model(output_location)
print(f"Saved to disk {output_location}")
