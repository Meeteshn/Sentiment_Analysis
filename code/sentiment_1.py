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

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA
torch.set_num_threads(4)  # Limit CPU threads to avoid overloading

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
        self.device = torch.device("cpu")  # Force CPU usage
        print(f"Using device for sarcasm model: {self.device}")
        
        try:
            # Load model configuration
            with open(os.path.join(model_dir, "model_config.json"), "r") as f:
                self.config = json.load(f)
            
            # Load tokenizer
            tokenizer_path = os.path.join(model_dir, "tokenizer")
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
            
            # Load weights with map_location to force CPU
            checkpoint = torch.load(
                os.path.join(model_dir, "best_model.pt"), 
                map_location=torch.device('cpu'),
                weights_only=True  # Add weights_only=True to address the warning
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.max_len = self.config.get("max_len", 64)
            self.test_accuracy = self.config.get("test_accuracy", "Unknown")
            
            print(f"Sarcasm detection model loaded (Test accuracy: {self.test_accuracy})")
            self.is_loaded = True
            
        except Exception as e:
            print(f"Error loading sarcasm model: {e}")
            self.is_loaded = False
    
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
        if not self.is_loaded:
            return {"error": "Model not loaded properly"}
        
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
        self.device = torch.device("cpu")  # Force CPU usage
        print(f"Using device for sentiment model: {self.device}")
        
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            
            # Load label mapping
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
            
            # Load training metadata if available
            try:
                metadata_path = os.path.join(os.path.dirname(model_path), "training_metadata.json")
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
                    print(f"Sentiment model loaded (Accuracy: {self.metadata.get('accuracy', 'Unknown')})")
            except FileNotFoundError:
                self.metadata = None
                print("Sentiment model loaded (no metadata found)")
                
            self.model.eval()
            self.is_loaded = True
            
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            self.is_loaded = False
    
    def predict(self, text):
        """Predict sentiment of a text"""
        if not self.is_loaded:
            return {"error": "Model not loaded properly"}
        
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
# COMBINED ANALYZER
###########################################
class TextReviewAnalyzer:
    def __init__(self, sarcasm_model_dir="saved_model", sentiment_model_dir="sentiment_model/final_model"):
        """Initialize both sarcasm and sentiment models"""
        self.sarcasm_detector = SarcasmDetector(sarcasm_model_dir)
        self.sentiment_analyzer = SentimentAnalyzer(sentiment_model_dir)
        
        # Check if models are loaded
        self.models_ready = (self.sarcasm_detector.is_loaded and self.sentiment_analyzer.is_loaded)
    
    def analyze_text(self, text):
        """Analyze text using both models"""
        if not self.models_ready:
            return {"error": "One or more models failed to load"}
        
        # Get predictions from both models
        sarcasm_result = self.sarcasm_detector.predict(text)
        sentiment_result = self.sentiment_analyzer.predict(text)
        
        # Combine results
        result = {
            "text": text,
            "sarcasm": sarcasm_result,
            "sentiment": sentiment_result,
            "combined_analysis": self._get_combined_analysis(sarcasm_result, sentiment_result)
        }
        
        return result
    
    def _get_combined_analysis(self, sarcasm_result, sentiment_result):
        """Generate combined analysis from both models' results"""
        is_sarcastic = sarcasm_result.get("is_sarcastic", False)
        sarcasm_confidence = sarcasm_result.get("confidence", 0)
        sentiment = sentiment_result.get("sentiment", "unknown")
        sentiment_confidence = sentiment_result.get("confidence", 0)
        
        # Build summary interpretation
        if is_sarcastic and sarcasm_confidence > 0.7:
            # High confidence sarcasm
            if sentiment == "positive" and sentiment_confidence > 0.7:
                interpretation = "This appears to be strong sarcasm with positive language (likely meaning the opposite)"
            elif sentiment == "negative" and sentiment_confidence > 0.7:
                interpretation = "This is sarcastic negativity, possibly exaggerating criticism"
            else:
                interpretation = "This text is strongly sarcastic"
        elif is_sarcastic and sarcasm_confidence > 0.5:
            # Medium confidence sarcasm
            interpretation = f"This text may be sarcastic with {sentiment} language"
        else:
            # Not sarcastic or low confidence
            interpretation = f"This text appears to express genuine {sentiment} sentiment"
        
        return {
            "interpretation": interpretation,
            "review_type": self._classify_review_type(sarcasm_result, sentiment_result)
        }
    
    def _classify_review_type(self, sarcasm_result, sentiment_result):
        """Classify the review type based on both analyses"""
        is_sarcastic = sarcasm_result.get("is_sarcastic", False)
        sarcasm_confidence = sarcasm_result.get("confidence", 0)
        sentiment = sentiment_result.get("sentiment", "unknown")
        sentiment_confidence = sentiment_result.get("confidence", 0)
        
        if is_sarcastic and sarcasm_confidence > 0.7:
            if sentiment == "positive" and sentiment_confidence > 0.6:
                return "Satirical criticism (positive language used sarcastically)"
            elif sentiment == "negative" and sentiment_confidence > 0.6:
                return "Sarcastic critique (negative and sarcastic)"
            else:
                return "Highly sarcastic review"
        elif sentiment == "positive" and sentiment_confidence > 0.7:
            return "Strong positive review"
        elif sentiment == "negative" and sentiment_confidence > 0.7:
            return "Strong negative review"
        elif sentiment == "positive":
            return "Moderate positive review"
        elif sentiment == "negative":
            return "Moderate negative review"
        else:
            return "Neutral or mixed review"

###########################################
# GUI APPLICATION
###########################################
class ReviewAnalyzerApp:
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
        
        # Initialize analyzer
        self.analyzer = None
        self.analyzer_initialized = False
        
        # Create UI
        self.create_ui()
        
        # Initialize analyzer in background
        self.root.after(100, self.initialize_analyzer)
    
    def create_ui(self):
        """Create the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Movie Review Analyzer", style="Title.TLabel")
        title_label.pack(pady=(0, 20))
        
        # Description
        desc_text = "Enter a movie review or any text to analyze both sentiment and detect sarcasm"
        desc_label = ttk.Label(main_frame, text=desc_text, wraplength=800)
        desc_label.pack(pady=(0, 15))
        
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
        self.status_label = ttk.Label(button_frame, text="Initializing models (using CPU)...")
        self.status_label.pack(side=tk.RIGHT)
        
        # Results frame
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Results section
        results_label = ttk.Label(results_frame, text="Analysis Results", style="Header.TLabel")
        results_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Create notebook/tabs for results
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Summary tab
        self.summary_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.summary_tab, text="Summary")
        
        # Sarcasm details tab
        self.sarcasm_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.sarcasm_tab, text="Sarcasm Analysis")
        
        # Sentiment details tab
        self.sentiment_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.sentiment_tab, text="Sentiment Analysis")
        
        # Initialize tabs with placeholder content
        self.init_results_tabs()
        
        # Footer
        footer_frame = ttk.Frame(main_frame)
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        footer_text = "This analyzer combines a BERT-based sarcasm detector and a DistilBERT sentiment analyzer"
        footer_label = ttk.Label(footer_frame, text=footer_text, foreground=self.neutral_color, font=("Arial", 9))
        footer_label.pack(side=tk.LEFT)
        
        # Set focus to input
        self.text_input.focus_set()
    
    def init_results_tabs(self):
        """Initialize the results tabs with placeholder content"""
        # Summary tab
        summary_content = ttk.Frame(self.summary_tab, padding=10)
        summary_content.pack(fill=tk.BOTH, expand=True)
        
        placeholder_text = "Enter a text and click 'Analyze Text' to see results"
        placeholder_label = ttk.Label(summary_content, text=placeholder_text)
        placeholder_label.pack(pady=20)
        
        # Store references to labels that will be updated
        self.summary_result_frame = summary_content
        
        # Sarcasm tab
        sarcasm_content = ttk.Frame(self.sarcasm_tab, padding=10)
        sarcasm_content.pack(fill=tk.BOTH, expand=True)
        
        placeholder_label = ttk.Label(sarcasm_content, text=placeholder_text)
        placeholder_label.pack(pady=20)
        
        self.sarcasm_result_frame = sarcasm_content
        
        # Sentiment tab
        sentiment_content = ttk.Frame(self.sentiment_tab, padding=10)
        sentiment_content.pack(fill=tk.BOTH, expand=True)
        
        placeholder_label = ttk.Label(sentiment_content, text=placeholder_text)
        placeholder_label.pack(pady=20)
        
        self.sentiment_result_frame = sentiment_content
    
    def initialize_analyzer(self):
        """Initialize the analyzer in the background"""
        try:
            # Get model paths (could be configurable)
            sarcasm_model_dir = "saved_model"
            sentiment_model_dir = "sentiment_model/final_model"
            
            # Check if directories exist
            if not os.path.exists(sarcasm_model_dir):
                self.status_label.config(text="Error: Sarcasm model not found")
                messagebox.showerror("Model Error", f"Sarcasm model directory not found: {sarcasm_model_dir}")
                return
                
            if not os.path.exists(sentiment_model_dir):
                self.status_label.config(text="Error: Sentiment model not found")
                messagebox.showerror("Model Error", f"Sentiment model directory not found: {sentiment_model_dir}")
                return
            
            # Initialize the analyzer
            self.analyzer = TextReviewAnalyzer(sarcasm_model_dir, sentiment_model_dir)
            
            if self.analyzer.models_ready:
                self.analyzer_initialized = True
                self.status_label.config(text="Models loaded successfully (CPU mode)")
                self.analyze_button.config(state=tk.NORMAL)
            else:
                self.status_label.config(text="Error: Models failed to initialize")
                messagebox.showerror("Model Error", "Failed to initialize one or more models. Check console for details.")
        
        except Exception as e:
            self.status_label.config(text="Error initializing models")
            messagebox.showerror("Initialization Error", f"Error initializing models: {str(e)}")
    
    def load_example(self, event):
        """Load an example review from the dropdown"""
        selected = self.example_var.get()
        if selected != "Select an example...":
            self.text_input.delete(1.0, tk.END)
            self.text_input.insert(tk.END, selected)
            # Reset the dropdown
            self.example_var.set("Select an example...")
    
    def clear_input(self):
        """Clear the input text"""
        self.text_input.delete(1.0, tk.END)
        
        # Clear result tabs
        for widget in self.summary_result_frame.winfo_children():
            widget.destroy()
        for widget in self.sarcasm_result_frame.winfo_children():
            widget.destroy()
        for widget in self.sentiment_result_frame.winfo_children():
            widget.destroy()
        
        # Add placeholder text
        placeholder_text = "Enter a text and click 'Analyze Text' to see results"
        ttk.Label(self.summary_result_frame, text=placeholder_text).pack(pady=20)
        ttk.Label(self.sarcasm_result_frame, text=placeholder_text).pack(pady=20)
        ttk.Label(self.sentiment_result_frame, text=placeholder_text).pack(pady=20)
    
    def analyze_text_event(self, event):
        """Handle Ctrl+Enter event to analyze text"""
        self.analyze_text()
    
    def analyze_text(self):
        """Analyze the input text"""
        # Check if analyzer is initialized
        if not self.analyzer_initialized:
            messagebox.showerror("Not Ready", "The models are still initializing. Please wait.")
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
            # Get analysis results
            results = self.analyzer.analyze_text(text)
            
            # Update the result tabs
            self.update_results(results)
            
            # Update status
            self.status_label.config(text="Analysis complete")
            
        except Exception as e:
            self.status_label.config(text="Error during analysis")
            messagebox.showerror("Analysis Error", f"Error during analysis: {str(e)}")
    
    def update_results(self, results):
        """Update the results tabs with the analysis"""
        # Clear existing results
        for widget in self.summary_result_frame.winfo_children():
            widget.destroy()
        for widget in self.sarcasm_result_frame.winfo_children():
            widget.destroy()
        for widget in self.sentiment_result_frame.winfo_children():
            widget.destroy()
        
        # Extract results
        sarcasm_result = results.get("sarcasm", {})
        sentiment_result = results.get("sentiment", {})
        combined_analysis = results.get("combined_analysis", {})
        
        # Update Summary tab
        self.update_summary_tab(sarcasm_result, sentiment_result, combined_analysis)
        
        # Update Sarcasm tab
        self.update_sarcasm_tab(sarcasm_result)
        
        # Update Sentiment tab
        self.update_sentiment_tab(sentiment_result)
        
        # Switch to Summary tab
        self.results_notebook.select(0)
    
    def update_summary_tab(self, sarcasm_result, sentiment_result, combined_analysis):
        """Update the summary tab with the analysis results"""
        # Create a summary frame
        frame = self.summary_result_frame
        
        # Overall interpretation
        interpretation = combined_analysis.get("interpretation", "No interpretation available")
        review_type = combined_analysis.get("review_type", "Unknown review type")
        
        # Create eye-catching summary header
        header_frame = ttk.Frame(frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        review_type_label = ttk.Label(header_frame, text=review_type, 
                                     font=("Arial", 16, "bold"), foreground=self.header_color)
        review_type_label.pack(pady=(5, 0))
        
        # Interpretation
        interp_frame = ttk.Frame(frame)
        interp_frame.pack(fill=tk.X, pady=(0, 20))
        
        interp_label = ttk.Label(interp_frame, text="Interpretation:", font=("Arial", 12, "bold"))
        interp_label.pack(anchor=tk.W)
        
        interp_text = ttk.Label(interp_frame, text=interpretation, wraplength=800)
        interp_text.pack(anchor=tk.W, pady=(5, 0))
        
        # Quick stats in a grid
        stats_frame = ttk.Frame(frame)
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Sarcasm detection result
        is_sarcastic = sarcasm_result.get("is_sarcastic", False)
        sarcasm_conf = sarcasm_result.get("confidence", 0) * 100
        
        sarcasm_style = "Sarcastic.TLabel" if is_sarcastic else "TLabel"
        sarcasm_frame = ttk.Frame(stats_frame)
        sarcasm_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        sarcasm_header = ttk.Label(sarcasm_frame, text="Sarcasm Detection", font=("Arial", 12, "bold"))
        sarcasm_header.pack(anchor=tk.W)
        
        sarcasm_result_text = f"{'Sarcastic' if is_sarcastic else 'Not Sarcastic'} ({sarcasm_conf:.1f}% confidence)"
        sarcasm_result_label = ttk.Label(sarcasm_frame, text=sarcasm_result_text, style=sarcasm_style)
        sarcasm_result_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Sentiment analysis result
        sentiment = sentiment_result.get("sentiment", "unknown")
        sentiment_conf = sentiment_result.get("confidence", 0) * 100
        
        sentiment_style = "Positive.TLabel" if sentiment == "positive" else "Negative.TLabel"
        sentiment_frame = ttk.Frame(stats_frame)
        sentiment_frame.pack(side=tk.LEFT)
        
        sentiment_header = ttk.Label(sentiment_frame, text="Sentiment Analysis", font=("Arial", 12, "bold"))
        sentiment_header.pack(anchor=tk.W)
        
        sentiment_result_text = f"{sentiment.capitalize()} ({sentiment_conf:.1f}% confidence)"
        sentiment_result_label = ttk.Label(sentiment_frame, text=sentiment_result_text, style=sentiment_style)
        sentiment_result_label.pack(anchor=tk.W, pady=(5, 0))
    
    def update_sarcasm_tab(self, sarcasm_result):
        """Update the sarcasm tab with detailed results"""
        frame = self.sarcasm_result_frame
        
        # Header
        header_label = ttk.Label(frame, text="Sarcasm Detection Results", style="Header.TLabel")
        header_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Main result
        is_sarcastic = sarcasm_result.get("is_sarcastic", False)
        confidence = sarcasm_result.get("confidence", 0) * 100
        
        result_style = "Sarcastic.TLabel" if is_sarcastic else "TLabel"
        result_label = ttk.Label(frame, 
                                text=f"Verdict: {'Sarcastic' if is_sarcastic else 'Not Sarcastic'}", 
                                style=result_style, 
                                font=("Arial", 14, "bold"))
        result_label.pack(anchor=tk.W, pady=(0, 5))
        
        conf_label = ttk.Label(frame, text=f"Confidence: {confidence:.1f}%")
        conf_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Feature details
        features = sarcasm_result.get("features", {})
        
        features_frame = ttk.Frame(frame)
        features_frame.pack(fill=tk.X, pady=(0, 15))
        
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
        explanation_frame = ttk.Frame(frame)
        explanation_frame.pack(fill=tk.X, pady=(10, 0))
        
        explanation_header = ttk.Label(explanation_frame, text="How Sarcasm is Detected:", font=("Arial", 12, "bold"))
        explanation_header.pack(anchor=tk.W, pady=(0, 5))
        
        explanation_text = (
            "The sarcasm detector combines BERT language understanding with features like sentiment "
            "and punctuation. Sarcasm often involves a contrast between positive/negative language "
            "and the intended meaning. Excessive punctuation (like '!!!' or '???') can also signal sarcasm."
        )
        
        explanation_label = ttk.Label(explanation_frame, text=explanation_text, wraplength=800)
        explanation_label.pack(anchor=tk.W)
    
    def update_sentiment_tab(self, sentiment_result):
        """Update the sentiment tab with detailed results"""
        frame = self.sentiment_result_frame
        
        # Header
        header_label = ttk.Label(frame, text="Sentiment Analysis Results", style="Header.TLabel")
        header_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Main result
        sentiment = sentiment_result.get("sentiment", "unknown")
        confidence = sentiment_result.get("confidence", 0) * 100
        
        result_style = "Positive.TLabel" if sentiment == "positive" else "Negative.TLabel"
        result_label = ttk.Label(frame, 
                                text=f"Sentiment: {sentiment.capitalize()}", 
                                style=result_style, 
                                font=("Arial", 14, "bold"))
        result_label.pack(anchor=tk.W, pady=(0, 5))
        
        conf_label = ttk.Label(frame, text=f"Confidence: {confidence:.1f}%")
        conf_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Probability distribution
        probs = sentiment_result.get("probabilities", {})
        
        if probs:
            probs_frame = ttk.Frame(frame)
            probs_frame.pack(fill=tk.X, pady=(0, 15))
            
            probs_header = ttk.Label(probs_frame, text="Probability Distribution:", font=("Arial", 12, "bold"))
            probs_header.pack(anchor=tk.W, pady=(0, 10))
            
            # Create a simple bar chart
            chart_frame = ttk.Frame(probs_frame)
            chart_frame.pack(fill=tk.X, pady=(0, 10))
            
            max_width = 400  # Maximum width of bars
            
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
                
                bar_canvas = tk.Canvas(bar_frame, width=max_width, height=20, bg=self.bg_color, highlightthickness=0)
                bar_canvas.pack(side=tk.LEFT, padx=(0, 10))
                bar_canvas.create_rectangle(0, 0, bar_width, 20, fill=bar_color, outline="")
        
        # Explanation
        explanation_frame = ttk.Frame(frame)
        explanation_frame.pack(fill=tk.X, pady=(10, 0))
        
        explanation_header = ttk.Label(explanation_frame, text="About Sentiment Analysis:", font=("Arial", 12, "bold"))
        explanation_header.pack(anchor=tk.W, pady=(0, 5))
        
        explanation_text = (
            "The sentiment analyzer uses a DistilBERT model fine-tuned on movie reviews to classify text as "
            "positive or negative. The confidence score indicates how certain the model is of its prediction. "
            "Higher confidence generally means more clearly positive or negative language in the text."
        )
        
        explanation_label = ttk.Label(explanation_frame, text=explanation_text, wraplength=800)
        explanation_label.pack(anchor=tk.W)

def main():
    # Make sure path to models exists
    if not os.path.exists("saved_model"):
        print("Warning: 'saved_model' directory not found. Sarcasm detection might not work.")
    
    if not os.path.exists("sentiment_model/final_model"):
        print("Warning: 'sentiment_model/final_model' directory not found. Sentiment analysis might not work.")
    
    # Create main window
    root = tk.Tk()
    
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
    app = ReviewAnalyzerApp(root)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()