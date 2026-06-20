import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("=" * 40)
print("SPAM SMS DETECTION MODEL")
print("=" * 40)

# --------------------------
# Load Dataset
# --------------------------
current_folder = os.path.dirname(__file__)
file_path = os.path.join(current_folder, "spam.csv")

data = pd.read_csv(file_path, encoding='latin-1')

# Keep required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Features and target
X = data['message']
y = data['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Text vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_features, y_train)

# Accuracy
y_pred = model.predict(X_test_features)
accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print(f"Accuracy: {accuracy * 100:.2f}%")

# --------------------------
# Prediction Loop
# --------------------------
while True:
    user_message = input(
        "\nEnter a message (or type 'exit' to stop): "
    )

    if user_message.lower() == "exit":
        print("\nProgram Closed")
        break

    message_feature = vectorizer.transform([user_message])

    prediction = model.predict(message_feature)[0]
    confidence = (
        model.predict_proba(message_feature).max()
        * 100
    )

    print("\nPrediction Result:")

    if prediction == 1:
        print("⚠️ This message is SPAM")
    else:
        print("✅ This message is NOT SPAM")

    print(f"Confidence: {confidence:.2f}%")