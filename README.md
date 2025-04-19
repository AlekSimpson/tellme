# TellMe

A CLI tool that translates natural language descriptions into precise shell commands.

## Description

TellMe is a command-line interface (CLI) tool that leverages AI to convert your natural language descriptions into ready-to-use shell commands. Just describe what you want to do, and TellMe will provide the appropriate command.

## Requirements

- Python 3.6+
- Ollama (with DeepSeek models installed)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tellme.git
cd tellme
```

2. Make sure you have Ollama installed with the DeepSeek models:
```bash
# Install Ollama according to https://ollama.com/download
# Then pull the DeepSeek models
ollama pull deepseek-r1:1.5b
ollama pull deepseek-r1:8b
```

3. Install the Python dependencies:
```bash
pip install ollama
```

4. Make the script executable (optional):
```bash
chmod +x main.py
# Create a symlink to use it from anywhere (optional)
ln -s $(pwd)/main.py /usr/local/bin/tellme
```

## Usage

### Basic Usage

```bash
python main.py "describe what command you need"
```

Or if you've set up the executable:

```bash
tellme "describe what command you need"
```

### Examples

```bash
# Find all Python files in the current directory
tellme "find all Python files in the current directory"

# Find and kill a process by name
tellme "find and kill the process named firefox"

# Compress a directory into a tar.gz file
tellme "compress the folder myproject into a tar.gz archive"

# Search for text in files
tellme "search for the word 'TODO' in all JavaScript files"

# Change file permissions
tellme "give read and write permissions to a file for the owner only"
```

## Features

- Translates natural language to shell commands
- Beautiful spinning dots animation while thinking
- Supports a variety of Unix/Linux command operations
- Designed for zsh shell
- Provides concise, accurate commands

## Configuration

You can modify the model size in the `tellme.py` file by changing the `MODELSIZE` variable:
```python
# Use "SMALL" for faster responses with the 1.5b model
# Use "LARGE" for more accurate responses with the 8b model
MODELSIZE = "LARGE"  
```

## License

[MIT License](LICENSE) 