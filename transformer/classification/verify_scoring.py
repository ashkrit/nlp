from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModel
import pickle
import sys
from sklearn.linear_model import LogisticRegression

model_id = "/Users/ashkrit/_tmp/model/finetuned-emotion"
classifier_finetune = pipeline("text-classification", model=model_id)


model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

model_path = "/Users/ashkrit/_tmp/model/distilbert-base-uncased-feature-model.pkl"
classifier_feature_trained = pickle.load(open(model_path, "rb"))


def features(input_text):
    encoded_new_input = tokenizer(text=input_text, return_tensors="pt")
    features = model(**encoded_new_input)[0].detach().numpy()[0]
    return features


print(f"Model {classifier_feature_trained}")
print(f"Model {classifier_finetune}")

id2label = {0: "sadness", 1: "joy", 2: "love",
            3: "anger", 4: "fear", 5: "surprise"}

print("How are you feeling ?>")
for line in sys.stdin:
    line = line.strip()
    result_finetune = classifier_finetune(line)
    result_feature = classifier_feature_trained.predict(features(line))

    print(f"fine tune {result_finetune}")
    print(f"feature {id2label[result_feature[0]]}")
    print("How are you feeling ? > ")
