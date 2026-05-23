# Twitter Sentiment Analysis using Machine Learning

## Project Overview
This project performs sentiment analysis on tweets using Machine Learning and Natural Language Processing (NLP).

The model classifies tweets into:
- Positive Sentiment
- Negative Sentiment

The project uses the Sentiment140 dataset and a Logistic Regression model.

---

## Features
- Data Cleaning and Preprocessing
- Stopword Removal
- TF-IDF Vectorization
- Logistic Regression Model
- Sentiment Prediction
- Streamlit Web Application
- Word Cloud Visualization

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- WordCloud
- Streamlit

---

## Project Structure

```bash
SentimentAnalysisProject/
│
├── app.py
├── sentiment_model.py
├── tweets.csv
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
└── venv/
```

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/SentimentAnalysisProject.git
```

### Step 2: Open Project Folder

```bash
cd SentimentAnalysisProject
```

### Step 3: Create Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

---

## Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## Run Model Training

```bash
python sentiment_model.py
```

This will create:
- model.pkl
- vectorizer.pkl

---

## Run Streamlit Application

```bash
streamlit run app.py
```

---

## Example Tweets

### Positive
```text
I love this mobile phone
```

### Negative
```text
This movie is very bad
```

---

## Machine Learning Model
- Logistic Regression

---

## Dataset
Sentiment140 Dataset

Link:
https://www.kaggle.com/datasets/kazanova/sentiment140

---

## Future Improvements
- Add Neutral Sentiment
- Use Deep Learning Models
- Deploy on Cloud
- Add Live Twitter API

---

## Author
Ashutosh