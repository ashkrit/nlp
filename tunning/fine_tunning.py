from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
import evaluate
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification,
    TrainingArguments, AdamW, get_scheduler)


checkpoint = "bert-base-uncased"

## Step 1 - Load Base model 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

## Step 2 - Prepare Traning Data
raw_datasets = load_dataset("glue", "mrpc")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)


## Step 3 - Config Model with optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
#num_training_steps = (int)(num_epochs * (len(train_dataloader)/4))  # to speed up

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
print(num_training_steps)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device) ##  bind it to device type 

progress_bar = tqdm(range(num_training_steps))

model_params = model.train()
print(model_params)

# Step 4 - Training
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


# Step 5 - Eval
metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

eval_result = metric.compute()
print("Eval Result ", eval_result)

# Step 6 - Save Model

time_stamp = datetime.now().strftime("%d%m%Y_%H%M%S")
model_path = "/Users/ashkrit/_model/" + checkpoint + "_" + time_stamp

tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)

print("Model Saved @", model_path)
