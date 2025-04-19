import re
import ollama as lama

# system_prompt = """
# You are a shell command assistant integrated into a CLI tool called "tellme". Your purpose is to translate natural language requests into precise shell commands.
# 
# RESPONSE FORMAT:
# 1. Return ONLY the exact command with minimal or no explanation.
# 2. Do NOT use markdown code blocks.
# 3. Do NOT include explanations.
# 4. Keep responses to a single line whenever possible.
# 5. Prefer practical, working solutions over theoretical explanations.
# 6. Should be SINGLE SHELL COMMAND.
# 
# RULES:
# 1. Assume the user is using zsh.
# 2. Assume the user is on linux.
# 3. NEVER explain what the command does
# 4. NEVER explain how the command works
# 5. ALWAYS prefer common utilities over obscure ones
# 6. Include short flags (-a) rather than long flags (--all) when possible
# 7. If multiple solutions exist, provide only the most efficient/common one
# 8. If a command requires sudo privileges, include sudo in the response
# 9. If a command might be destructive, prefix with a brief "CAUTION: " marker
# 10. If you don't know a command for a specific operation, say "Unknown operation" rather than guessing
# 11. ONLY GIVE ONE FINAL ANSWER COMMAND. NEVER GIVE MULTIPLE OPTIONS.
# 12. Keep the answer BRIEF; preferably only output the single command.
# 13. Answer should NOT be longer than 15-20 words.
# """

system_prompt = """
Your task is to provide a bash shell command that matches the user's description of what the command should do.
Your response should be ONLY a SINGLE bash shell command.
The command should be accurate.
The command should do what the user describes and nothing else.
Do not explain the command.
Do not give tutorials on how to accompolish the task.
Keep the answers simple and brief.
Do not make up CLI tools.
If the command does not exist or is impossible, simply respond with "Unknown Command."
GIVE COMMANDS ONLY FOR LINUX. NO OTHER OPERATING SYSTEMS.
YOU DO NOT NEED TO EXPLAIN OR ACTUALLY PERFORM THE TASK YOURSELF.

Examples of Good Responses:
---------------------------------
User: download only the audio from a youtube video given the youtube video's link
Response: 
```bash
youtube-dl --extract-audio --format bestaudio https://www.youtube.com/watch?v=VIDEO_ID
```

Replace `VIDEO_ID` with the actual video ID from the URL.

---------------------------------
User: list all files in a directory that have the .json file descriptor
Response: ls -r *.json
"""

# Generate text
def tellme(prompt):
    remove_think_section = lambda text: re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    print("Thinking...")
    full_prompt = lambda prompt_: f"{system_prompt}\ngive me the command to: {prompt}"
    server_response = lama.generate(model='deepseek-r1:8b', prompt=full_prompt(prompt), options={
        'temperature': 0.1
    })
    prompt_response = remove_think_section(server_response['response'])
    return prompt_response

