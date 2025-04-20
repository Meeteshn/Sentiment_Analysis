import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocess_kgptalkie as ps
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments
from transformers import default_data_collator

# ---- Install Dependencies (run only if needed) ----
def install_dependencies():
    print("Installing required dependencies...")
    os.system("pip install -U transformers accelerate datasets bertviz umap-learn")
    os.system("pip install spacy beautifulsoup4 textblob mlxtend")
    os.system("pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git --upgrade --force-reinstall")
    os.system("python -m spacy download en_core_web_sm")

# Uncomment the line below if you need to install dependencies
# install_dependencies()

# ---- Check GPU Availability ----
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# ---- Custom Dataset Class ----
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx])

        encoding = self.tokenizer(text, truncation=True, padding="max_length",
                                 max_length=self.max_len, return_tensors="pt")

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

# ---- Load and Preprocess Dataset ----
def load_and_preprocess_data(data_path=None, sample_size=10000):
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        print("Using default IMDB dataset from GitHub...")
        df = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/IMDB-Dataset.csv")
    
    # Sample the dataset if needed
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    print(f"Dataset loaded with {len(df)} samples")
    print("First few rows:")
    print(df.head())
    
    # Feature Engineering
    df['word_counts'] = df['review'].apply(lambda x: len(str(x).split()))
    df['char_counts'] = df['review'].apply(lambda x: len(str(x)))
    df['avg_wordlength'] = df.apply(lambda row: row['char_counts'] / row['word_counts'] 
                                  if row['word_counts'] > 0 else 0, axis=1)
    df['stopwords_counts'] = df['review'].apply(lambda x: sum(1 for word in str(x).split() 
                                              if word in ['the', 'is', 'in', 'and', 'to']))
    
    # Text Preprocessing
    df['review'] = df['review'].str.lower()
    df['review'] = df['review'].apply(lambda x: ps.remove_html_tags(x))
    
    # Display word count distribution
    plt.figure(figsize=(10, 6))
    df['word_counts'].value_counts().sort_index().plot(kind='bar')
    plt.title('Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.savefig('word_count_distribution.png')
    plt.close()
    
    return df

# ---- Compute Metrics for Evaluation ----
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {'accuracy': acc, "f1": f1}

# ---- Main Training Function ----
def train_sentiment_model(data_path=None, 
                         model_save_path="sentiment_model", 
                         epochs=1, 
                         batch_size=16, 
                         learning_rate=2e-5,
                         max_len=512,
                         pretrained_model="distilbert-base-uncased"):
    
    # Load and preprocess data
    df = load_and_preprocess_data(data_path)
    
    # Prepare labels
    label2id = {'positive': 1, 'negative': 0}
    id2label = {1: 'positive', 0: 'negative'}
    
    # Save mapping for inference
    import json
    os.makedirs(model_save_path, exist_ok=True)
    with open(os.path.join(model_save_path, "label_mapping.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f)
    
    # Split data into text and labels
    X = df['review'].tolist()
    y = df['sentiment'].map(label2id).tolist()
    
    # Split into train/test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(train_texts)} samples")
    print(f"Test set: {len(test_texts)} samples")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model, 
        num_labels=len(label2id)
    ).to(device)
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_len)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_len)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(model_save_path, "checkpoints"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(model_save_path, "logs"),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator
    )
    
    # Train the model
    print("Starting model training...")
    trainer.train()
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save the results to a file
    with open(os.path.join(model_save_path, "eval_results.txt"), "w") as f:
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")
    
    # Save the model and tokenizer
    model_path = os.path.join(model_save_path, "final_model")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    
    print(f"Model and tokenizer saved to {model_path}")
    
    # Save some training metadata
    with open(os.path.join(model_save_path, "training_metadata.json"), "w") as f:
        metadata = {
            "pretrained_model": pretrained_model,
            "max_sequence_length": max_len,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_samples": len(train_texts),
            "test_samples": len(test_texts),
            "accuracy": eval_results.get("eval_accuracy", 0),
            "f1_score": eval_results.get("eval_f1", 0),
            "training_device": device
        }
        json.dump(metadata, f, indent=4)
    
    return model_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model")
    parser.add_argument("--data_path", type=str, default=None, 
                        help="Path to the dataset CSV file (default: use IMDB dataset from GitHub)")
    parser.add_argument("--model_save_path", type=str, default="sentiment_model", 
                        help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=1, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--max_len", type=int, default=512, 
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--pretrained_model", type=str, default="distilbert-base-uncased", 
                        help="Pretrained model to fine-tune")
    parser.add_argument("--sample_size", type=int, default=10000, 
                        help="Number of samples to use from the dataset (default: 10000)")
    
    args = parser.parse_args()
    
    train_sentiment_model(
        data_path=args.data_path,
        model_save_path=args.model_save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_len=args.max_len,
        pretrained_model=args.pretrained_model
    )