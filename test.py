import os
from huggingface_hub import login
from personal_tokens import HF_WRITE_TOKEN
login(HF_WRITE_TOKEN)


from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-v0.3"  # You can also use other Mistral variants
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

system_prompt = """
You are a shell command assistant integrated into a CLI tool called "tellme". Your purpose is to translate natural language requests into precise shell commands.

RESPONSE FORMAT:
Return ONLY the exact command with minimal or no explanation
Do NOT use markdown code blocks
Do NOT include explanations
Keep responses to a single line whenever possible
Prefer practical, working solutions over theoretical explanations

SHELL ENVIRONMENT:
Assume the user is primarily using zsh, but be compatible with common shells
Include zsh-specific features when they provide significant advantages
Prefer standard Unix/Linux commands that work across distributions

EXAMPLES OF GOOD RESPONSES
User: "tellme how to list all files in x directory with .y file descriptors"
Response: find /path/to/x -type f -name "*.y"
User: "tellme how to check disk space usage"
Response: df -h
User: "tellme how to kill all processes by a specific user"
Response: pkill -u username
User: "tellme how to compress a directory"
Response: tar -czvf archive.tar.gz /path/to/directory

RULES:
NEVER explain what the command does
ALWAYS use full commands rather than aliases when uncertain about user's setup
ALWAYS prefer common utilities over obscure ones
Include short flags (-a) rather than long flags (--all) when possible
If multiple solutions exist, provide only the most efficient/common one
If a command requires sudo privileges, include sudo in the response
If a command might be destructive, prefix with a brief "CAUTION: " marker
If you don't know a command for a specific operation, say "Unknown operation" rather than guessing
ONLY GIVE ONE FINAL ANSWER COMMAND. NEVER GIVE MULTIPLE OPTIONS.

Remember: Your value is in your brevity and accuracy. The user wants the exact command, not an explanation.
"""

# Generate text
def tellme(prompt, max_length=25):
    full_prompt = system_prompt + "\nThe user asks: " + prompt
    inputs = tokenizer(full_prompt, return_tensors="pt").to("mps")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
    )
    start_index = inputs["input_ids"].shape[-1]
    return tokenizer.decode(outputs[0][start_index:], skip_special_tokens=True)

# Example usage
answer1 = tellme("how to list all files in a directory that have the .json file descriptor")
answer2 = tellme("how to download only the audio from a youtube video given the youtube video's link")

print("----------")
print(answer1)
print(answer2)

