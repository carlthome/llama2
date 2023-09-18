# Llama 2 usage example with 🤗 Transformers

A tiny example of running Llama 2 locally.

## Usage

```sh
# Create a virtual environment and activate it.
python -m venv .venv
.venv/bin/activate

# Install dependencies.
pip install -r requirements.txt

# Login to access gated model weights.
huggingface-cli login

# Run the example.
python predict.py
```
