import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.3"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "My favourite condiment is"
model_inputs = tokenizer([prompt], return_tensors="pt").to("mps")
model.to("mps")
generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)

tokenizer.batch_decode(generated_ids)[0]
