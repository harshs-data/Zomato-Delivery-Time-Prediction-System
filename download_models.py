import gdown
import os
from pathlib import Path

# Absolute paths (works on both Windows and Linux/Render)
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Your verified File ID
file_id = "1Z2cRlddS8YQENM6fFVa-wGzfuXuJ-Svw"
output = str(MODELS_DIR / "stacking_regressor.joblib")

# 1. Create the models directory if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)

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

