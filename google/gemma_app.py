from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it")

system_prompt = "You are a helpful assistant."

while True:
    text = input("question:")

    messages = [

        {"role": "user", "content": system_prompt + "\n\n" + text}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("Prompt " + prompt)

    input = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    outputs = model.generate(input_ids = input.to(model.device), max_length=200, 
                             do_sample=True, top_k=50, top_p=0.95, temperature=0.8)
    print("Answer -> ")
    print(tokenizer.decode(outputs[0]))
