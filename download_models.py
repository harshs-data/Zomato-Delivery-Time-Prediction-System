import gdown
import os
from pathlib import Path

# Your verified File ID
file_id = "1Z2cRlddS8YQENM6fFVa-wGzfuXuJ-Svw"
output = "models/stacking_regressor.joblib"

# 1. Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# 2. Download the file if it's not already there
if not os.path.exists(output):
    print("Downloading large model file from Google Drive...")
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        # This handles the download directly into your models folder
        gdown.download(url, output, quiet=False)
        print("\nDownload complete ✓")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        # If it fails, the app will crash later, so we want to see this error
else:
    print("Model file already exists. Skipping download.")
