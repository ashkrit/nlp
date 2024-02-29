from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os 


os.environ ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", # 'google/gemma-2b-it
                                    )
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

chat = [
    { "role": "user", "content": "What is the difference between LLaMAs, Alpacas, and Vicunas" },
]

prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print(prompt)

inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
outputs = model.generate(input_ids=inputs.to(model.device),max_new_tokens=512)

text = tokenizer.decode(outputs[0],skip_special_tokens=True, clean_up_tokenization_spaces=True)

print(text)
