import pandas as pd

columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

data = pd.read_csv(
    "tweets.csv",
    encoding='latin-1',
    names=columns
)

print(data.head())

# Keep Required Columns Only
data = data[['sentiment', 'text']]

print(data.head())

# Convert Sentiment Labels
data['sentiment'] = data['sentiment'].replace(4, 1)

print(data['sentiment'].value_counts())

# Clean Tweet Text
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)

    words = text.lower().split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

data['text'] = data['text'].apply(clean_text)

print(data.head())

# Convert Text to Numbers
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(data['text'])

y = data['sentiment']


# Split Dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Machine Learning Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

# Predict Results
predictions = model.predict(X_test)

# Check Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Add Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print(cm)

# Create Word Cloud Visualization
from wordcloud import WordCloud
import matplotlib.pyplot as plt

all_words = " ".join(data['text'])

wordcloud = WordCloud(
    width=800,
    height=400
).generate(all_words)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# Save Model
import pickle

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
