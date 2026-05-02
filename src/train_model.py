import pandas as pd
import os
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading cleaned data...")
    cleaned_data_path = DATA_DIR / "cleaned_data.csv"
    if not cleaned_data_path.exists():
        print(f"Error: {cleaned_data_path} not found.")
        return

    df = pd.read_csv(str(cleaned_data_path))
    df = df.dropna(subset=['cleaned_text'])
    
    X = df['cleaned_text']
    y = df['sentiment']
    
    print("Splitting data 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Vectorizing text (TF-IDF max_features=5000)...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'MultinomialNB': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_acc = 0
    best_name = ""
    
    accuracies = {}
    
    print("Training models...")
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        accuracies[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    print(f"\nAuto-selected best model: {best_name} with accuracy {best_acc:.4f}")
    
    # Save best model and vectorizer
    joblib.dump(best_model, str(MODELS_DIR / "sentiment_model.pkl"))
    joblib.dump(vectorizer, str(MODELS_DIR / "vectorizer.pkl"))
    
    # Generate and save outputs for best model
    preds = best_model.predict(X_test_vec)
    report = classification_report(y_test, preds)
    with open(str(OUTPUTS_DIR / "classification_report.txt"), 'w') as f:
        f.write(report)
        
    cm = confusion_matrix(y_test, preds, labels=best_model.classes_)
    
    # Using dark styling for plots to match dashboard vibe eventually
    plt.style.use('dark_background')
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / "confusion_matrix.png"), transparent=True)
    plt.close()
    
    plt.figure(figsize=(10,6))
    colors = sns.color_palette("Blues_d", len(accuracies))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
    plt.title('Model Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies.values()):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / "model_comparison.png"), transparent=True)
    plt.close()
    
    print("Training outputs and models saved successfully.")

if __name__ == "__main__":
    main()