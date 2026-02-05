"""Helper to log in to HuggingFace"""
from huggingface_hub import login

print("Logging in to HuggingFace...")
print("Paste your token")
print()

try:
    login()
    print("\nLogin successful!")
    print("You can now run sam3_pipeline.py")
except Exception as e:
    print(f"\nLogin failed: {e}")
