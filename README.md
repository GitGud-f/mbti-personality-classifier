# MBTI Personality Type Classifier

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This project classifies Myers-Briggs Type Indicator (MBTI) personality types from social media posts using natural language processing (NLP). Designed as a well-structured Python package, it follows software engineering best practices.

## 🎯 Features

- Clean, modular codebase organized as a proper Python package (`src/` layout)
<!-- - Support for classical (TF-IDF + Logistic Regression) and transformer-based models (e.g., DistilBERT) -->
<!-- - Configurable preprocessing, training, and evaluation pipelines -->
<!-- - Reproducible experiments via YAML configuration -->
<!-- - Unit tests and command-line interface (CLI) -->
<!-- - Detailed README and professional project structure -->

## 📊 Dataset

The model is trained on the [MBTI Kaggle dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type), which contains:
- **8,600 rows** having posts collected from online forums
- Each row labeled with one of **16 MBTI types** (e.g., `INFJ`, `ENTP`)
- Raw text includes social media-style writing (emojis, slang, typos)

> 💡 *Note: Due to licensing, the dataset is not included in this repo. Download it from Kaggle and place it in `data/raw/`.*

<!-- ## 🧠 Model Options

| Approach              | Technique                     | Pros                              |
|-----------------------|-------------------------------|-----------------------------------|
| **Baseline**          | TF-IDF + Logistic Regression  | Fast, interpretable, lightweight |
| **Advanced (opt.)**   | Fine-tuned DistilBERT         | Higher accuracy, contextual understanding |

*(Default: TF-IDF baseline for simplicity and speed)* -->

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/GitGud-f/mbti-personality-classifier.git
cd mbti-personality-classifier
``` 

### 2. Set up environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Add your data
Download the [MBTI dataset from Kaggle](https://www.kaggle.com/datasets/datasnaek/mbti-type) and save it as:
```
data/raw/mbti_1.csv
```

### 4. Train the model -todo
```bash
python scripts/train_model.py
```

### 5. Make a prediction - todo
```bash
python scripts/predict.py --text "I love deep conversations and quiet nights."
# Output: INFJ
```

## 📁 Project Structure

```
personality-classifier/
├── data/                 # Raw & processed datasets
├── models/               # Saved model artifacts
├── src/
│   └── personality_classifier/  # Core package
│       ├── data/         # Loading & preprocessing
│       ├── features/     # Text vectorization
│       ├── models/       # Training & inference
│       └── utils/        # Helpers & metrics
├── scripts/              # CLI entry points
├── tests/                # Unit tests
└── config/               # Experiment configurations
```

## 🧪 Testing - todo

Run unit tests to ensure pipeline integrity:
```bash
pytest tests/
```

## 🛠️ Configuration - todo

Edit `config/config.yaml` to:
- Change model hyperparameters
- Toggle preprocessing options
- Set random seeds for reproducibility

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙌 Acknowledgements

- Dataset: [Kaggle MBTI Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type)
