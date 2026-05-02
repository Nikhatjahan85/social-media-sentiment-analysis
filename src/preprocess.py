import pandas as pd
import os
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import preprocess_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

def main():
    social_media_data_path = DATA_DIR / "social_media_data.csv"
    print(f"Loading {social_media_data_path}...")
    
    if not social_media_data_path.exists():
        print(f"Data not found at {social_media_data_path}. Please run data/generate_dataset.py first.")
        return

    df = pd.read_csv(str(social_media_data_path))
    
    print("Applying preprocessing pipeline...")
    df['cleaned_text'] = df['text'].apply(preprocess_pipeline)
    
    cleaned_data_path = DATA_DIR / "cleaned_data.csv"
    df.to_csv(str(cleaned_data_path), index=False)
    
    print("Shape of cleaned data:", df.shape)
    print("Value counts:\n", df['sentiment'].value_counts())
    print("Sample rows:\n", df[['text', 'cleaned_text', 'sentiment']].head())

if __name__ == "__main__":
    main()