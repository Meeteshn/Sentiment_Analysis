import os
import sys
import json
import torch
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForSequenceClassification
import re
from PIL import Image, ImageTk

# Make sure VADER lexicon is downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

###########################################
# SARCASM DETECTOR MODEL
###########################################
class SarcasmDetector:
    def __init__(self, model_dir="saved_model"):
        """Initialize the sarcasm detector model"""
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        self.model = None
        self.tokenizer = None
        self.config = None
        
        # Try to load config only (lightweight)
        try:
            with open(os.path.join(model_dir, "model_config.json"), "r") as f:
                self.config = json.load(f)
                self.max_len = self.config.get("max_len", 64)
                self.test_accuracy = self.config.get("test_accuracy", "Unknown")
        except Exception as e:
            print(f"Error loading sarcasm model config: {e}")
    
    def load_model(self):
        """Load the model into GPU memory only when needed"""
        if self.is_loaded:
            return True
            
        try:
            # Clear GPU memory first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load tokenizer
            tokenizer_path = os.path.join(self.model_dir, "tokenizer")
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            
            # Initialize model architecture
            from torch import nn
            class ContextAwareSarcasmDetector(nn.Module):
                def __init__(self, model_name="bert-base-uncased", custom_feat_dim=3, hidden_dim=128, num_labels=2):
                    super(ContextAwareSarcasmDetector, self).__init__()
                    self.bert = BertModel.from_pretrained(model_name)
                    bert_output_dim = self.bert.config.hidden_size
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
            
            # Create model with config
            self.model = ContextAwareSarcasmDetector(
                model_name="bert-base-uncased",
                custom_feat_dim=self.config.get("custom_feat_dim", 3),
                hidden_dim=self.config.get("hidden_dim", 128),
                num_labels=2
            )
            
            # Load weights
            checkpoint = torch.load(
                os.path.join(self.model_dir, "best_model.pt"), 
                map_location=self.device,
                weights_only=True
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Sarcasm detection model loaded to {self.device} (Test accuracy: {self.test_accuracy})")
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading sarcasm model: {e}")
            self.is_loaded = False
            return False
    
    def unload_model(self):
        """Unload the model to free GPU memory"""
        if not self.is_loaded:
            return
            
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False
        print("Sarcasm model unloaded from GPU")
    
    def clean_text(self, text):
        """Clean text for sarcasm detection"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9!\?\.,'\"\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def get_custom_features(self, text):
        """Compute custom features for sarcasm detection"""
        if not isinstance(text, str):
            return np.array([0.0, 0, 0], dtype=np.float32)
        
        sentiment = sia.polarity_scores(text)['compound']
        exclam_count = text.count('!')
        question_count = text.count('?')
        return np.array([sentiment, exclam_count, question_count], dtype=np.float32)
    
    def predict(self, text):
        """Predict if a text is sarcastic"""
        # Load model if not already loaded
        if not self.load_model():
            return {"error": "Model failed to load"}
        
        # Clean text
        clean_input = self.clean_text(text)
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            clean_input,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True
        )
        
        # Get custom features
        custom_feats = torch.tensor(self.get_custom_features(clean_input)).unsqueeze(0)
        
        # Move tensors to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        custom_feats = custom_feats.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, custom_feats)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()
        
        # Return prediction details
        result = {
            "is_sarcastic": bool(pred_class),
            "confidence": confidence,
            "label": "Sarcastic" if pred_class == 1 else "Not Sarcastic",
            "features": {
                "sentiment_score": float(self.get_custom_features(clean_input)[0]),
                "exclamation_marks": int(self.get_custom_features(clean_input)[1]),
                "question_marks": int(self.get_custom_features(clean_input)[2])
            }
        }
        
        return result

###########################################
# SENTIMENT ANALYSIS MODEL
###########################################
class SentimentAnalyzer:
    def __init__(self, model_path="sentiment_model/final_model"):
        """Initialize the sentiment analyzer model"""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.metadata = None
        
        # Try to load metadata only (lightweight)
        try:
            mapping_path = os.path.join(os.path.dirname(model_path), "label_mapping.json")
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
                self.id2label = mapping["id2label"]
                # Convert string keys back to integers
                self.id2label = {int(k): v for k, v in self.id2label.items()}
        except FileNotFoundError:
            print(f"Warning: Label mapping file not found at {mapping_path}")
            print("Using default mapping: {0: 'negative', 1: 'positive'}")
            self.id2label = {0: 'negative', 1: 'positive'}
    
    def load_model(self):
        """Load the model into GPU memory only when needed"""
        if self.is_loaded:
            return True
            
        try:
            # Clear GPU memory first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device)
            
            # Load training metadata if available
            try:
                metadata_path = os.path.join(os.path.dirname(self.model_path), "training_metadata.json")
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
                    accuracy = self.metadata.get('accuracy', 'Unknown')
                    print(f"Sentiment model loaded to {self.device} (Accuracy: {accuracy})")
            except FileNotFoundError:
                self.metadata = None
                print(f"Sentiment model loaded to {self.device} (no metadata found)")
                
            self.model.eval()
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            self.is_loaded = False
            return False
    
    def unload_model(self):
        """Unload the model to free GPU memory"""
        if not self.is_loaded:
            return
            
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False
        print("Sentiment model unloaded from GPU")
    
    def predict(self, text):
        """Predict sentiment of a text"""
        # Load model if not already loaded
        if not self.load_model():
            return {"error": "Model failed to load"}
        
        # Handle empty or invalid input
        if not isinstance(text, str) or not text.strip():
            return {"sentiment": "neutral", "confidence": 0.0}
        
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Get predicted class and confidence
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()
                
                # Get probabilities for all classes
                probs_dict = {self.id2label[i]: float(probs[0, i].item()) 
                             for i in range(len(self.id2label))}
                    
                result = {
                    "sentiment": self.id2label[pred_class],
                    "confidence": confidence,
                    "probabilities": probs_dict
                }
                    
                return result
        except Exception as e:
            print(f"Error in sentiment prediction: {e}")
            return {"sentiment": "unknown", "confidence": 0.0, "error": str(e)}

###########################################
# GUI APPLICATION
###########################################
class ToggleableReviewAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Review Analyzer")
        self.root.geometry("1000x700")
        self.root.minsize(900, 600)
        
        # Set theme
        self.style = ttk.Style()
        self.style.theme_use('clam')  # 'clam', 'alt', 'default', 'classic'
        
        # Define colors
        self.bg_color = "#f0f0f0"
        self.header_color = "#3a7ca5"
        self.accent_color = "#2c3e50"
        self.text_color = "#333333"
        self.highlight_color = "#e74c3c"
        self.positive_color = "#27ae60"
        self.negative_color = "#c0392b"
        self.neutral_color = "#7f8c8d"
        
        # Configure theme colors
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.text_color)
        self.style.configure("TButton", background=self.accent_color, foreground="white")
        self.style.configure("Header.TLabel", font=("Arial", 16, "bold"), foreground=self.header_color)
        self.style.configure("Title.TLabel", font=("Arial", 20, "bold"), foreground=self.header_color)
        self.style.configure("Result.TLabel", font=("Arial", 12))
        self.style.configure("Positive.TLabel", foreground=self.positive_color, font=("Arial", 12, "bold"))
        self.style.configure("Negative.TLabel", foreground=self.negative_color, font=("Arial", 12, "bold"))
        self.style.configure("Sarcastic.TLabel", foreground=self.highlight_color, font=("Arial", 12, "bold"))
        
        # Initialize model flags
        self.current_model = "none"  # "sentiment", "sarcasm", or "none"
        
        # Initialize analyzers (but don't load models yet)
        self.sarcasm_detector = SarcasmDetector("saved_model")
        self.sentiment_analyzer = SentimentAnalyzer("sentiment_model/final_model")
        
        # Set GPU availability message
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            self.gpu_message = f"Using GPU: {gpu_name}"
        else:
            self.gpu_message = "GPU not available, using CPU"
        
        # Create UI
        self.create_ui()
    
    def create_ui(self):
        """Create the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Movie Review Analyzer", style="Title.TLabel")
        title_label.pack(pady=(0, 10))
        
        # GPU Info
        gpu_label = ttk.Label(main_frame, text=self.gpu_message, 
                             font=("Arial", 10, "italic"), foreground=self.neutral_color)
        gpu_label.pack(pady=(0, 15))
        
        # Description
        desc_text = "Enter a movie review and select which type of analysis to perform"
        desc_label = ttk.Label(main_frame, text=desc_text, wraplength=800)
        desc_label.pack(pady=(0, 15))
        
        # Analysis Type Selection
        model_frame = ttk.Frame(main_frame)
        model_frame.pack(fill=tk.X, pady=(0, 15))
        
        model_label = ttk.Label(model_frame, text="Analysis Type:", font=("Arial", 12, "bold"))
        model_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Radio buttons for model selection
        self.model_var = tk.StringVar(value="sentiment")
        
        sentiment_radio = ttk.Radiobutton(
            model_frame, text="Sentiment Analysis", 
            variable=self.model_var, value="sentiment",
            command=self.on_model_change
        )
        sentiment_radio.pack(side=tk.LEFT, padx=(0, 15))
        
        sarcasm_radio = ttk.Radiobutton(
            model_frame, text="Sarcasm Detection", 
            variable=self.model_var, value="sarcasm",
            command=self.on_model_change
        )
        sarcasm_radio.pack(side=tk.LEFT)
        
        # Input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Input label
        input_label = ttk.Label(input_frame, text="Enter text to analyze:", style="Header.TLabel")
        input_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Text input
        self.text_input = scrolledtext.ScrolledText(input_frame, height=6, width=80, font=("Arial", 12))
        self.text_input.pack(fill=tk.BOTH, expand=True)
        self.text_input.bind("<Control-KeyRelease-Return>", self.analyze_text_event)  # Ctrl+Enter shortcut
        
        # Example reviews dropdown
        examples_frame = ttk.Frame(main_frame)
        examples_frame.pack(fill=tk.X, pady=(5, 10))
        
        example_label = ttk.Label(examples_frame, text="Try an example: ")
        example_label.pack(side=tk.LEFT)
        
        self.example_var = tk.StringVar()
        example_options = [
            "Select an example...",
            "This movie was absolutely incredible! I loved every minute of it.",
            "Worst film ever. Complete waste of time and money.",
            "Yeah, this movie was sooo amazing. I especially loved the 2-hour long boring intro.",
            "The special effects were good, but the plot was confusing.",
            "Oh great, another superhero movie. Just what the world needed."
        ]
        
        example_dropdown = ttk.Combobox(examples_frame, textvariable=self.example_var, values=example_options, width=50)
        example_dropdown.current(0)
        example_dropdown.pack(side=tk.LEFT, padx=5)
        example_dropdown.bind("<<ComboboxSelected>>", self.load_example)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Analyze button
        self.analyze_button = ttk.Button(button_frame, text="Analyze Text", command=self.analyze_text)
        self.analyze_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear button
        clear_button = ttk.Button(button_frame, text="Clear", command=self.clear_input)
        clear_button.pack(side=tk.LEFT)
        
        # Status indicator
        self.status_label = ttk.Label(button_frame, text="Ready - Select analysis type and enter text")
        self.status_label.pack(side=tk.RIGHT)
        
        # Results frame
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Results section
        results_label = ttk.Label(results_frame, text="Analysis Results", style="Header.TLabel")
        results_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Results content frame
        self.results_content = ttk.Frame(results_frame, padding=10, relief="solid", borderwidth=1)
        self.results_content.pack(fill=tk.BOTH, expand=True)
        
        # Add placeholder text
        placeholder_text = "Select an analysis type and enter text to analyze"
        self.placeholder_label = ttk.Label(self.results_content, text=placeholder_text)
        self.placeholder_label.pack(pady=20)
        
        # Footer
        footer_frame = ttk.Frame(main_frame)
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        footer_text = "Note: Only one model is loaded at a time to maximize GPU memory usage"
        footer_label = ttk.Label(footer_frame, text=footer_text, foreground=self.neutral_color, font=("Arial", 9))
        footer_label.pack(side=tk.LEFT)
        
        # Set focus to input
        self.text_input.focus_set()
        
        # Pre-load the initial model
        self.on_model_change()
    
    def on_model_change(self):
        """Handle model change and load the selected model"""
        new_model = self.model_var.get()
        
        if new_model == self.current_model:
            return  # No change needed
        
        # Unload current model
        if self.current_model == "sentiment":
            self.sentiment_analyzer.unload_model()
        elif self.current_model == "sarcasm":
            self.sarcasm_detector.unload_model()
        
        # Update status
        if new_model == "sentiment":
            self.status_label.config(text="Loading sentiment analysis model...")
            self.root.update_idletasks()
            success = self.sentiment_analyzer.load_model()
            if success:
                self.status_label.config(text="Sentiment analysis model ready")
            else:
                self.status_label.config(text="Error loading sentiment model")
                messagebox.showerror("Model Error", "Failed to load sentiment model")
                
        elif new_model == "sarcasm":
            self.status_label.config(text="Loading sarcasm detection model...")
            self.root.update_idletasks()
            success = self.sarcasm_detector.load_model()
            if success:
                self.status_label.config(text="Sarcasm detection model ready")
            else:
                self.status_label.config(text="Error loading sarcasm model")
                messagebox.showerror("Model Error", "Failed to load sarcasm model")
        
        # Update current model
        self.current_model = new_model
    
    def load_example(self, event):
        """Load an example review from the dropdown"""
        selected = self.example_var.get()
        if selected != "Select an example...":
            self.text_input.delete(1.0, tk.END)
            self.text_input.insert(tk.END, selected)
            # Reset the dropdown
            self.example_var.set("Select an example...")
    
    def clear_input(self):
        """Clear the input text and results"""
        self.text_input.delete(1.0, tk.END)
        
        # Clear results
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        # Add placeholder text
        placeholder_text = "Select an analysis type and enter text to analyze"
        self.placeholder_label = ttk.Label(self.results_content, text=placeholder_text)
        self.placeholder_label.pack(pady=20)
    
    def analyze_text_event(self, event):
        """Handle Ctrl+Enter event to analyze text"""
        self.analyze_text()
    
    def analyze_text(self):
        """Analyze the input text"""
        # Check which model is selected
        model_type = self.model_var.get()
        if model_type not in ["sentiment", "sarcasm"]:
            messagebox.showerror("Selection Error", "Please select an analysis type")
            return
        
        # Get text from input
        text = self.text_input.get(1.0, tk.END).strip()
        
        if not text:
            messagebox.showinfo("Empty Input", "Please enter some text to analyze.")
            return
        
        # Update status
        self.status_label.config(text="Analyzing...")
        self.root.update_idletasks()
        
        try:
            # Run appropriate analysis
            if model_type == "sentiment":
                result = self.sentiment_analyzer.predict(text)
                self.display_sentiment_results(result)
            else:  # sarcasm
                result = self.sarcasm_detector.predict(text)
                self.display_sarcasm_results(result)
            
            # Update status
            self.status_label.config(text="Analysis complete")
            
        except Exception as e:
            self.status_label.config(text="Error during analysis")
            messagebox.showerror("Analysis Error", f"Error during analysis: {str(e)}")
    
    def display_sentiment_results(self, result):
        """Display sentiment analysis results"""
        # Clear existing results
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        # Check for errors
        if "error" in result:
            error_label = ttk.Label(self.results_content, text=f"Error: {result['error']}")
            error_label.pack(pady=20)
            return
        
        # Header frame
        header_frame = ttk.Frame(self.results_content)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        sentiment = result.get("sentiment", "unknown")
        confidence = result.get("confidence", 0) * 100
        
        # Main result
        result_style = "Positive.TLabel" if sentiment == "positive" else "Negative.TLabel"
        result_label = ttk.Label(header_frame, 
                                text=f"Sentiment: {sentiment.capitalize()}", 
                                style=result_style, 
                                font=("Arial", 18, "bold"))
        result_label.pack(anchor=tk.W, pady=(0, 5))
        
        conf_label = ttk.Label(header_frame, text=f"Confidence: {confidence:.1f}%")
        conf_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Divider
        ttk.Separator(self.results_content, orient='horizontal').pack(fill='x', pady=10)
        
        # Probability distribution
        probs = result.get("probabilities", {})
        
        if probs:
            probs_frame = ttk.Frame(self.results_content)
            probs_frame.pack(fill=tk.X, pady=(15, 15))
            
            probs_header = ttk.Label(probs_frame, text="Probability Distribution:", font=("Arial", 12, "bold"))
            probs_header.pack(anchor=tk.W, pady=(0, 10))
            
            # Create a simple bar chart
            chart_frame = ttk.Frame(probs_frame)
            chart_frame.pack(fill=tk.X, pady=(0, 10))
            
            max_width = 500  # Maximum width of bars
            
            for label, prob in probs.items():
                bar_frame = ttk.Frame(chart_frame)
                bar_frame.pack(fill=tk.X, pady=5)
                
                # Label
                label_text = f"{label.capitalize()}: {prob*100:.1f}%"
                label_width = 150
                label_widget = ttk.Label(bar_frame, text=label_text, width=20)
                label_widget.pack(side=tk.LEFT)
                
                # Bar
                bar_width = int(max_width * prob)
                bar_color = self.positive_color if label == "positive" else self.negative_color
                
                bar_canvas = tk.Canvas(bar_frame, width=max_width, height=25, bg=self.bg_color, highlightthickness=0)
                bar_canvas.pack(side=tk.LEFT, padx=(0, 10))
                bar_canvas.create_rectangle(0, 0, bar_width, 25, fill=bar_color, outline="")
        
        # Explanation
        explanation_frame = ttk.Frame(self.results_content)
        explanation_frame.pack(fill=tk.X, pady=(15, 0))
        
        explanation_header = ttk.Label(explanation_frame, text="About Sentiment Analysis:", font=("Arial", 12, "bold"))
        explanation_header.pack(anchor=tk.W, pady=(0, 5))
        
        explanation_text = (
            "The sentiment analyzer uses a DistilBERT model fine-tuned on movie reviews to classify text as "
            "positive or negative. The confidence score indicates how certain the model is of its prediction. "
            "Higher confidence generally means more clearly positive or negative language in the text."
        )
        
        explanation_label = ttk.Label(explanation_frame, text=explanation_text, wraplength=800)
        explanation_label.pack(anchor=tk.W)
    
    def display_sarcasm_results(self, result):
        """Display sarcasm detection results"""
        # Clear existing results
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        # Check for errors
        if "error" in result:
            error_label = ttk.Label(self.results_content, text=f"Error: {result['error']}")
            error_label.pack(pady=20)
            return
        
        # Header frame
        header_frame = ttk.Frame(self.results_content)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        is_sarcastic = result.get("is_sarcastic", False)
        confidence = result.get("confidence", 0) * 100
        
        # Main result
        result_style = "Sarcastic.TLabel" if is_sarcastic else "TLabel"
        verdict_text = "Sarcastic" if is_sarcastic else "Not Sarcastic"
        result_label = ttk.Label(header_frame, 
                                text=f"Verdict: {verdict_text}", 
                                style=result_style, 
                                font=("Arial", 18, "bold"))
        result_label.pack(anchor=tk.W, pady=(0, 5))
        
        conf_label = ttk.Label(header_frame, text=f"Confidence: {confidence:.1f}%")
        conf_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Divider
        ttk.Separator(self.results_content, orient='horizontal').pack(fill='x', pady=10)
        
        # Feature details
        features = result.get("features", {})
        
        if features:
            features_frame = ttk.Frame(self.results_content)
            features_frame.pack(fill=tk.X, pady=(15, 15))
            
            features_header = ttk.Label(features_frame, text="Contributing Factors:", font=("Arial", 12, "bold"))
            features_header.pack(anchor=tk.W, pady=(0, 10))
            
            # Sentiment score
            sentiment_score = features.get("sentiment_score", 0)
            sentiment_label = ttk.Label(features_frame, 
                                      text=f"Sentiment Score: {sentiment_score:.2f} " + 
                                           f"({'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'})")
            sentiment_label.pack(anchor=tk.W, pady=(0, 5))
            
            # Punctuation
            exclam_count = features.get("exclamation_marks", 0)
            question_count = features.get("question_marks", 0)
            
            punct_label = ttk.Label(features_frame, 
                                   text=f"Exclamation Marks: {exclam_count}   |   Question Marks: {question_count}")
            punct_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Explanation
        explanation_frame = ttk.Frame(self.results_content)
        explanation_frame.pack(fill=tk.X, pady=(15, 0))
        
        explanation_header = ttk.Label(explanation_frame, text="How Sarcasm is Detected:", font=("Arial", 12, "bold"))
        explanation_header.pack(anchor=tk.W, pady=(0, 5))
        
        explanation_text = (
            "The sarcasm detector combines BERT language understanding with features like sentiment "
            "and punctuation. Sarcasm often involves a contrast between positive/negative language "
            "and the intended meaning. Excessive punctuation (like '!!!' or '???') can also signal sarcasm."
        )
        
        explanation_label = ttk.Label(explanation_frame, text=explanation_text, wraplength=800)
        explanation_label.pack(anchor=tk.W)
        
        # Interpretation based on confidence
        interp_frame = ttk.Frame(self.results_content)
        interp_frame.pack(fill=tk.X, pady=(15, 0))
        
        if is_sarcastic and confidence > 85:
            interp_text = "This text is very likely to be sarcastic."
        elif is_sarcastic and confidence > 70:
            interp_text = "This text shows strong signs of sarcasm."
        elif is_sarcastic and confidence > 60:
            interp_text = "This text may be sarcastic, but with moderate confidence."
        elif not is_sarcastic and confidence > 85:
            interp_text = "This text shows no signs of sarcasm."
        elif not is_sarcastic and confidence > 70:
            interp_text = "This text is probably not sarcastic."
        else:
            interp_text = "The sarcasm detection is uncertain for this text."
        
        interp_style = "Sarcastic.TLabel" if is_sarcastic and confidence > 70 else "TLabel"
        interp_label = ttk.Label(interp_frame, text=interp_text, style=interp_style, font=("Arial", 12))
        interp_label.pack(anchor=tk.W)


def main():
    # Check model directories
    if not os.path.exists("saved_model"):
        print("Warning: 'saved_model' directory not found. Sarcasm detection might not work.")
    
    if not os.path.exists("sentiment_model/final_model"):
        print("Warning: 'sentiment_model/final_model' directory not found. Sentiment analysis might not work.")
    
    # Check GPU availability
    if torch.cuda.is_available():
        try:
            # Try allocating a small tensor to verify GPU works
            test_tensor = torch.zeros(1, device='cuda')
            print(f"GPU is available: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        except Exception as e:
            print(f"GPU is available but encountered an error: {e}")
    else:
        print("GPU is not available. Using CPU.")
    
    # Create main window
    root = tk.Tk()
    root.title("Movie Review Analyzer (GPU Mode)")
    
    # Set app icon if available
    try:
        # Try to set icon
        icon_path = "icon.png"  # Create your own icon file
        if os.path.exists(icon_path):
            icon = ImageTk.PhotoImage(file=icon_path)
            root.iconphoto(True, icon)
    except:
        pass  # Icon setting is not critical
    
    # Create app
    app = ToggleableReviewAnalyzerApp(root)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()