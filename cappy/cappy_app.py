from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('btan2/cappy-large')
cappy = AutoModelForSequenceClassification.from_pretrained('btan2/cappy-large')

instruction = """
What label best describes this news article?
Carlyle Looks Toward Commercial Aerospace (Reuters) Reuters - Private investment firm Carlyle Group,
which has a reputation for making well-timed and occasionally
controversial plays in the defense industry, has quietly placed
its bets on another part of the market.
"""
response = 'business'

inputs = tokenizer([(instruction, response), ], return_tensors='pt')
score = cappy(**inputs).logits[0][0].item()

print(score)