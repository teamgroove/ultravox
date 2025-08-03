from huggingface_hub import list_models

print("Available Ultravox models:")
models = list(list_models(author='fixie-ai', search='ultravox', sort="downloads", direction=-1))
for m in models[:15]:
    print(f"  {m.id}")
