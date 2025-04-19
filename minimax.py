from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, QuantoConfig, GenerationConfig

# load hf config
hf_config = AutoConfig.from_pretrained("MiniMaxAI/MiniMax-Text-01", trust_remote_code=True)

# quantization config, int8 is recommended
quantization_config =  QuantoConfig(
            weights="int8",
            modules_to_not_convert=[
                "lm_head",
                "embed_tokens",
            ] + [f"model.layers.{i}.coefficient" for i in range(hf_config.num_hidden_layers)]
            + [f"model.layers.{i}.block_sparse_moe.gate" for i in range(hf_config.num_hidden_layers)]
        )

# assume 8 GPUs
world_size = 8
layers_per_device = hf_config.num_hidden_layers // world_size
# set device map
device_map = {
    'model.embed_tokens': 'mps:0',
    'model.norm': f'mps:{world_size - 1}',
    'lm_head': f'mps:{world_size - 1}'
}
for i in range(world_size):
    for j in range(layers_per_device):
        device_map[f'model.layers.{i * layers_per_device + j}'] = f'mps:{i}'

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-Text-01")
prompt = "Hello!"
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant created by MiniMax based on MiniMax-Text-01 model."}]},
    {"role": "user", "content": [{"type": "text", "text": prompt}]},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
# tokenize and move to device
model_inputs = tokenizer(text, return_tensors="pt").to("mps")

# load bfloat16 model, move to device, and apply quantization
quantized_model = AutoModelForCausalLM.from_pretrained(
    "MiniMaxAI/MiniMax-Text-01",
    torch_dtype="bfloat16",
    device_map=device_map,
    quantization_config=quantization_config,
    trust_remote_code=True,
    offload_buffers=True,
)

# generate response
generation_config = GenerationConfig(
    max_new_tokens=20,
    eos_token_id=200020,
    use_cache=True,
)
generated_ids = quantized_model.generate(**model_inputs, generation_config=generation_config)
print(f"generated_ids: {generated_ids}")
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
