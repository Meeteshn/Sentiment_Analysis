import json
import re
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import nltk
import os
import argparse
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

# Download VADER lexicon if needed
nltk.download('vader_lexicon', quiet=True)

###############################################################################
# 1. LOAD DATA & TEXT PREPROCESSING
###############################################################################
def load_json_data(filepath):
    """
    Loads the dataset from the given JSON file,
    returning a Pandas DataFrame with columns: ['headline', 'is_sarcastic'].
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line as JSON: {line[:50]}...")
                continue
    
    df = pd.DataFrame(data)
    
    # Check if required columns exist
    if "headline" not in df.columns or "is_sarcastic" not in df.columns:
        print("Warning: Required columns 'headline' and 'is_sarcastic' not found in the dataset.")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    df = df[["headline", "is_sarcastic"]]  # Keep only necessary columns
    return df

def clean_text(text):
    """
    Cleans text by converting to lowercase and removing unwanted characters
    (while keeping punctuation).
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9!\?\.,'\"\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

###############################################################################
# 2. CUSTOM FEATURE EXTRACTION
###############################################################################
sia = SentimentIntensityAnalyzer()  # Initialize VADER sentiment analyzer

def get_custom_features(text):
    """
    Computes custom features:
      - compound sentiment score,
      - count of exclamation marks,
      - count of question marks.
    Returns a numpy array of shape (3,).
    """
    if not isinstance(text, str):
        return np.array([0.0, 0, 0], dtype=np.float32)
    
    sentiment = sia.polarity_scores(text)['compound']
    exclam_count = text.count('!')
    question_count = text.count('?')
    return np.array([sentiment, exclam_count, question_count], dtype=np.float32)

###############################################################################
# 3. CUSTOM DATASET
###############################################################################
class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Handle non-string inputs
        if not isinstance(text, str):
            text = ""
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt", 
            truncation=True
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        custom_feats = get_custom_features(text)
        custom_feats = torch.tensor(custom_feats, dtype=torch.float32)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "custom_feats": custom_feats,
            "label": torch.tensor(label, dtype=torch.long)
        }

###############################################################################
# 4. CONTEXT-AWARE SARCASTIC DETECTOR MODEL
###############################################################################
class ContextAwareSarcasmDetector(nn.Module):
    def __init__(self, model_name="bert-base-uncased", custom_feat_dim=3, hidden_dim=128, num_labels=2):
        super(ContextAwareSarcasmDetector, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        bert_output_dim = self.bert.config.hidden_size  # typically 768
        self.fc = nn.Linear(bert_output_dim + custom_feat_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(hidden_dim, num_labels)
    
    def forward(self, input_ids, attention_mask, custom_feats):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        combined = torch.cat((cls_output, custom_feats), dim=1)
        x = self.dropout(combined)
        x = torch.relu(self.fc(x))
        x = self.dropout(x)
        logits = self.out(x)
        return logits

###############################################################################
# 5. TRAINING & EVALUATION FUNCTIONS
###############################################################################
def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        custom_feats = batch["custom_feats"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, custom_feats)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    preds = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            custom_feats = batch["custom_feats"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids, attention_mask, custom_feats)
            predictions = torch.argmax(outputs, dim=1)
            
            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
    return accuracy_score(true_labels, preds), classification_report(true_labels, preds)

###############################################################################
# 6. MAIN SCRIPT
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Train a sarcasm detection model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSON dataset")
    parser.add_argument("--output_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--max_len", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print(f"Loading data from {args.data_path}")
    df = load_json_data(args.data_path)
    
    if df is None or len(df) == 0:
        print("Error: Could not load data or dataset is empty. Please check your file path and format.")
        return
    
    print(f"Loaded {len(df)} headlines.")
    
    # Add extra sarcastic examples for better training
    extra_sarcasm = [
        "When people ask me stupid questions, it is my legal obligation to give a sarcastic remark.",
        "I'm not saying I hate you, what I'm saying is that you are literally the Monday of my life.",
        "Silence is golden. Duct tape is silver.",
        "I am busy right now, can I ignore you some other time?",
        "Find your patience before I lose mine.",
        "It's okay if you don't like me. Not everyone has good taste.",
        "Do you think God gets stoned? I think so… look at the platypus.",
        "Light travels faster than sound. This is why some people appear bright until they speak.",
        "If you find me offensive, then I suggest you quit finding me.",
        "Sarcasm is the body's natural defense against stupidity.",
        "I love sarcasm. It's like punching people in the face but with words.",
        "Life's good, you should get one.",
        "Cancel my subscription because I don't need your issues.",
        "I clapped because it's finished, not because I like it.",
        "If I had a dollar for every smart thing you say, I'll be poor.",
        "I'm sorry while you were talking I was trying to figure out where the hell you got the idea I cared.",
        "No, you don't have to repeat yourself. I was ignoring you the first time.",
        "Sarcasm is the secret language that everyone uses when they want to say something mean to your face.",
        "Unless your name is Google stop acting like you know everything.",
        "You know the difference between a tornado and divorce in the South? Nothing! Someone's losing a trailer, number one.",
        "I don't have the energy to pretend to like you today.",
        "I'm sorry I hurt your feelings when I called you stupid. I really thought you already knew.",
        "I never forget a face, but in your case, I'll be glad to make an exception.",
        "Sarcasm–the ability to insult idiots without them realizing it.",
        "If you think nobody cares if you're alive, try missing a couple of car payments.",
        "My imaginary friend says that you need a therapist.",
        "Well at least your mom thinks you're pretty.",
        "Sometimes I need what only you can provide: your absence.",
        "Just because I don't care doesn't mean I don't understand.",
        "Why do they call it rush hour when nothing moves?",
        "My neighbor's diary says that I have boundary issues.",
        "I would like to apologize to anyone I have not offended yet. Please be patient. I will get to you shortly.",
        "Don't worry about what people think. They don't do it very often.",
        "If at first, you don't succeed, skydiving is not for you.",
        "People say that laughter is the best medicine… your face must be curing the world.",
        "When I ask for directions, please don't use words like 'East.'",
        "Sometimes the amount of self-control it takes to not say what's on my mind is so immense, I need a nap afterward.",
        "The stuff you heard about me is a lie. I'm way worse.",
        "Me pretending to listen should be enough for you."
    ]
    extra_df = pd.DataFrame({
        "headline": extra_sarcasm,
        "is_sarcastic": [1] * len(extra_sarcasm)
    })
    df = pd.concat([df, extra_df], ignore_index=True)
    
    # Clean the text
    df["clean_headline"] = df["headline"].apply(clean_text)
    
    # Check for empty text after cleaning
    empty_count = df["clean_headline"].str.strip().eq("").sum()
    if empty_count > 0:
        print(f"Warning: {empty_count} headlines are empty after cleaning.")
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the data
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["is_sarcastic"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["is_sarcastic"], random_state=42)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    # Save the test set for later evaluation
    test_df.to_csv(os.path.join(args.output_dir, "test_data.csv"), index=False)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Save the tokenizer for inference
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    
    # Create datasets
    train_dataset = SarcasmDataset(train_df["clean_headline"].tolist(), train_df["is_sarcastic"].tolist(), tokenizer, args.max_len)
    val_dataset = SarcasmDataset(val_df["clean_headline"].tolist(), val_df["is_sarcastic"].tolist(), tokenizer, args.max_len)
    test_dataset = SarcasmDataset(test_df["clean_headline"].tolist(), test_df["is_sarcastic"].tolist(), tokenizer, args.max_len)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size)
    
    # Initialize model
    try:
        model = ContextAwareSarcasmDetector(
            model_name="bert-base-uncased", 
            custom_feat_dim=3, 
            hidden_dim=args.hidden_dim, 
            num_labels=2
        )
        model.to(device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        try:
            train_loss = train_model(model, train_loader, optimizer, device)
            print(f"  Training Loss: {train_loss:.4f}")
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"  WARNING: GPU out of memory during training, trying to free some memory...")
                torch.cuda.empty_cache()
                print(f"  Consider reducing batch size or model size")
                continue
            else:
                print(f"  Error during training: {e}")
                continue
        
        # Evaluate on validation set
        try:
            val_acc, val_report = evaluate_model(model, val_loader, device)
            print(f"  Validation Accuracy: {val_acc:.4f}")
            print(val_report)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"  WARNING: GPU out of memory during validation, trying to free some memory...")
                torch.cuda.empty_cache()
                continue
            else:
                print(f"  Error during validation: {e}")
                continue
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save model
            model_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'custom_feat_dim': 3,
                'hidden_dim': args.hidden_dim,
            }, model_path)
            print(f"  Saved best model to {model_path}")
    
    # Load the best model for final evaluation
    try:
        checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"))
        model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
        print("No saved model found. Skipping test evaluation.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Final evaluation on test set
    try:
        test_acc, test_report = evaluate_model(model, test_loader, device)
        print("\n=== Test Set Evaluation ===")
        print(f"Accuracy: {test_acc:.4f}")
        print(test_report)
    except Exception as e:
        print(f"Error during test evaluation: {e}")
        test_acc = 0.0
    
    # Save model config information
    with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
        json.dump({
            "max_len": args.max_len,
            "custom_feat_dim": 3,
            "hidden_dim": args.hidden_dim,
            "test_accuracy": float(test_acc)
        }, f)
    
    print(f"Model and configuration saved to {args.output_dir}")

if __name__ == "__main__":
    main()