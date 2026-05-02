import joblib
import pandas as pd
import os
from pathlib import Path
import random
from datetime import datetime, timedelta
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import preprocess_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def predict_sentiment(text, model, vectorizer):
    cleaned = preprocess_pipeline(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    
    probs = model.predict_proba(vec)[0]
    max_prob = max(probs)
    return pred, max_prob

def predict_batch(csv_path, model, vectorizer):
    df = pd.read_csv(str(csv_path))
    df['cleaned_text'] = df['text'].astype(str).apply(preprocess_pipeline)
    vecs = vectorizer.transform(df['cleaned_text'])
    preds = model.predict(vecs)
    probs = model.predict_proba(vecs).max(axis=1)
    df['predicted_sentiment'] = preds
    df['confidence'] = probs
    return df

def generate_sample_predictions():
    sentiment_model_path = MODELS_DIR / "sentiment_model.pkl"
    if not sentiment_model_path.exists():
        print("Model not found. Train the model first.")
        return

    model = joblib.load(str(sentiment_model_path))
    vectorizer = joblib.load(str(MODELS_DIR / "vectorizer.pkl"))
    
    print("--- Demo test 5 sample texts ---")
    samples = [
        "This new dashboard is completely intuitive and runs very fast. Excellent work!",
        "I am facing terrible timeout errors constantly. Please fix this bug.",
        "The system status remains unchanged after the recent update.",
        "Really reliable performance lately, amazing app.",
        "Experiencing latency issues during peak hours, very frustrating."
    ]
    
    for s in samples:
        pred, conf = predict_sentiment(s, model, vectorizer)
        print(f"Text: '{s}'\n-> Prediction: {pred} ({conf:.2f})\n")
        
    sample_predictions_path = OUTPUTS_DIR / "sample_predictions.csv"
    print(f"Generating sample predictions dataset ({sample_predictions_path})...")
    
    data = []
    end_time = datetime.now()
    
    try:
        df_pool = pd.read_csv(str(DATA_DIR / "social_media_data.csv"))
        sample_texts = df_pool['text'].sample(25, replace=True).tolist()
    except FileNotFoundError:
        sample_texts = samples * 5  # fallback if data not there
    
    for i, text in enumerate(sample_texts):
        pred, conf = predict_sentiment(text, model, vectorizer)
        
        # simulated time in the last day
        ts = (end_time - timedelta(minutes=random.randint(1, 1440))).strftime("%H:%M:%S.%f")[:-3]
        src_id = f"SRC-{random.randint(100, 999)}"
        frag = text[:50] + "..." if len(text) > 50 else text
        
        if pred == 'POSITIVE':
            score = round(random.uniform(0.71, 0.99), 3)
        elif pred == 'NEGATIVE':
            score = round(random.uniform(0.01, 0.29), 3)
        else:
            score = round(random.uniform(0.40, 0.60), 3)
            
        data.append({
            'timestamp': ts,
            'source_id': src_id,
            'content_fragment': frag,
            'score': score,
            'status': pred
        })
        
    df_out = pd.DataFrame(data)
    df_out = df_out.sort_values(by='timestamp', ascending=False)
    df_out.to_csv(str(sample_predictions_path), index=False)
    print(f"Saved {sample_predictions_path} successfully.")

if __name__ == "__main__":
    generate_sample_predictions()