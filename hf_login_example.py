import os
from huggingface_hub import login

# Get token from environment variable
token = os.getenv("HF_TOKEN")
if not token:
    print("Please set the HF_TOKEN environment variable with your Hugging Face token")
    print("Example: export HF_TOKEN=your_token_here")
    exit(1)

login(token=token)
print("Successfully logged into HuggingFace!")
