# Twitter Sentiment Analysis

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-stable-brightgreen)

A Python-based NLP project for **classifying tweets** as neutral or hate speech using traditional ML models like:
- Random Forest
- Logistic Regression
- Decision Tree
- XGBoost

This project includes:
- Preprocessing with NLTK
- Text vectorization with CountVectorizer
- Model training + evaluation using F1-score
- Test prediction export with timestamped `.csv`

---

## Project Structure
```bash
Twitter_sentiment_analysis
├── twitter_sentiment.py         # Main training + prediction pipeline
├── requirements.txt             # Python dependencies
├── submission_*.csv             # Exported test results (auto-generated)
└── .gitignore                   # Clean repo from CSVs, venv, cache
```

---

## How to Run
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it (Windows)
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the main pipeline
python twitter_sentiment.py
```

---

## Sample Output
```csv
id,tweet,label
31963,Just finished a great movie,0
31964,@user you are the worst ever,1
```

---

## Release
Check the [latest release here](https://github.com/Dinesh2226/Twitter-sentiment-analysis/releases)

---

## License
This project is licensed under the **MIT License**.
