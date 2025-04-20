# Sentiment_Analysis

# Movie Review Analysis Tool

A dual-purpose text analysis application that combines sentiment analysis and sarcasm detection for movie reviews and other text inputs.

## Overview

This project integrates two powerful NLP models to provide comprehensive text analysis:

1. **Sentiment Analysis**: A DistilBERT-based classifier trained on movie reviews to determine if text expresses positive or negative sentiment.

2. **Sarcasm Detection**: A context-aware BERT model that detects sarcastic statements by analyzing both semantic content and linguistic patterns.

The application provides a user-friendly GUI where users can enter text and get instant analysis using either model.
![Screenshot 2025-04-20 135214](https://github.com/user-attachments/assets/b6272204-a4db-4dc1-a80f-490da801777c)



## Features

- **Dual Analysis Capabilities**: Analyze text for sentiment or sarcasm with a simple toggle
- **GPU Acceleration**: Optimized for GPU usage with efficient memory management
- **User-Friendly Interface**: Clean design with intuitive controls
- **Pre-loaded Examples**: Test with sample reviews to see how the models perform
- **Detailed Analysis Results**: Confidence scores and feature breakdowns
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Installation

1. Clone the repository:
```bash
git clone [https://github.com/Meeteshn/Sentiment_Analysis.git]
cd Sentiment_analysis
```

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

  ## Train models: 
  The trained models aren't included in this repository due to their large size (~1.5GB).

  




## Datasets
--------


The `dataset` folder contains all datasets 


* **Sentiment Analysis**: Uses the IMDB movie reviews dataset which is not requred as, in my code it is diorectly included from the web
* **Sarcasm Detection**: Uses a dataset of news headlines which is included in `dataset` folder

## Model Performance
-----------------

* **Sentiment Analysis**:  
  ~92% accuracy on IMDB movie reviews dataset

* **Sarcasm Detection**:  
  ~85% accuracy on news headlines dataset with additional custom examples  

## Troubleshooting
---------------

* **CUDA memory errors**:  
  If you encounter CUDA out-of-memory errors with the GPU version, try using the CPU version instead.

* **Module not found errors**:  
  Make sure you've installed all requirements and are running the scripts from the project root directory.

## Future Improvements
-------------------

* Add combined analysis mode that intelligently merges both models' outputs  
* Implement batch processing for analyzing multiple reviews at once  
* Add support for more languages  
* Create a web-based interface  

## License
-------

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
---------------

* The sentiment analysis model is based on DistilBERT  
* The sarcasm detection model is based on BERT with custom feature extraction  
* The sentiment training code can use the IMDB dataset from GitHub  


