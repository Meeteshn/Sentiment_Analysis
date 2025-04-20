import json
import re
import torch
import argparse
import os
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if needed
nltk.download('vader_lexicon')

# Initialize VADER
sia = SentimentIntensityAnalyzer()

def clean_text(text):
    """
    Cleans text by converting to lowercase and removing unwanted characters
    (while keeping punctuation).
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9!\?\.,'\"\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_custom_features(text):
    """
    Computes custom features:
      - compound sentiment score,
      - count of exclamation marks,
      - count of question marks.
    Returns a numpy array of shape (3,).
    """
    sentiment = sia.polarity_scores(text)['compound']
    exclam_count = text.count('!')
    question_count = text.count('?')
    return np.array([sentiment, exclam_count, question_count], dtype=np.float32)

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

def load_model(model_dir):
    """
    Loads the saved model and its configuration
    """
    # Load configuration
    with open(os.path.join(model_dir, "model_config.json"), "r") as f:
        config = json.load(f)
    
    # Initialize model with the same architecture
    model = ContextAwareSarcasmDetector(
        model_name="bert-base-uncased",
        custom_feat_dim=config["custom_feat_dim"],
        hidden_dim=config["hidden_dim"],
        num_labels=2
    )
    
    # Load weights
    checkpoint = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config

def predict_sarcasm(text, model, tokenizer, max_len, device):
    """
    Predicts if a given text is sarcastic or not
    """
    # Clean text
    clean_input = clean_text(text)
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        clean_input,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True
    )
    
    # Get input tensors
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    custom_feats = torch.tensor(get_custom_features(clean_input)).unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, custom_feats)
        pred = torch.argmax(outputs, dim=1).item()
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0][pred].item()
    
    return pred, confidence

def main():
    parser = argparse.ArgumentParser(description="Predict sarcasm in text using a trained model")
    parser.add_argument("--model_dir", type=str, default="./saved_model", 
                        help="Directory where the model is saved")
    parser.add_argument("--interactive", action="store_true", 
                        help="Run in interactive mode")
    parser.add_argument("--text", type=str, default=None, 
                        help="Text to predict (when not in interactive mode)")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer_path = os.path.join(args.model_dir, "tokenizer")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    # Load model
    model, config = load_model(args.model_dir)
    model.to(device)
    print(f"Model loaded from {args.model_dir}")
    print(f"Test accuracy of the model: {config['test_accuracy']:.4f}")
    
    if args.interactive:
        print("\nSarcasm Detection Interactive Mode")
        print("Enter 'quit' to exit")
        
        while True:
            try:
                user_input = input("\nEnter a sentence: ").strip()
                if user_input.lower() == 'quit':
                    print("Exiting interactive session.")
                    break
                
                pred, confidence = predict_sarcasm(user_input, model, tokenizer, config["max_len"], device)
                
                if pred == 1:
                    print(f"Prediction: Sarcastic (Confidence: {confidence:.2f})")
                else:
                    print(f"Prediction: Not Sarcastic (Confidence: {confidence:.2f})")
                    
            except KeyboardInterrupt:
                print("\nExiting interactive session.")
                break
    
    elif args.text:
        pred, confidence = predict_sarcasm(args.text, model, tokenizer, config["max_len"], device)
        
        if pred == 1:
            print(f"Prediction: Sarcastic (Confidence: {confidence:.2f})")
        else:
            print(f"Prediction: Not Sarcastic (Confidence: {confidence:.2f})")
    
    else:
        print("No text provided. Run with --interactive flag or provide text with --text option.")

if __name__ == "__main__":
    main()