import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# sample training data
texts = [
    "I love this movie",
    "This is amazing",
    "Very good product",
    "I hate this",
    "Worst experience",
    "Not good"
]

labels = [1, 1, 1, 0, 0, 0]  # 1 = positive, 0 = negative

# convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# train model
model = MultinomialNB()
model.fit(X, labels)

# save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model created successfully!")