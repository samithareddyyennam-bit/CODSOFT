import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import os

# Get current folder path
current_folder = os.path.dirname(__file__)

# Full path of spam.csv
file_path = os.path.join(current_folder, "spam.csv")

# Load dataset
data = pd.read_csv(file_path, encoding="latin-1")

# Keep useful columns
data = data[['v1', 'v2']]

# Rename columns
data.columns = ['label', 'message']

# Convert labels into numbers
# ham = 0, spam = 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['message'])

# Labels
y = data['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("===================================")
print("SPAM SMS DETECTION MODEL")
print("===================================")
print("Model trained successfully!")
print("Accuracy:", accuracy)

# Continuous message prediction
while True:
    user_message = input("\nEnter a message (or type 'exit' to stop): ")

    # Stop program
    if user_message.lower() == 'exit':
        print("Program Closed")
        break

    # Convert message
    message_vector = vectorizer.transform([user_message])

    # Predict
    result = model.predict(message_vector)

    # Confidence score
    probability = model.predict_proba(message_vector)

    print("\nPrediction Result:")

    if result[0] == 1:
        print("This message is SPAM")
    else:
        print("This message is NOT SPAM")

    print(f"Confidence: {max(probability[0]) * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Spam", "Spam"]
)

display.plot()

plt.title("Spam SMS Detection Confusion Matrix")
plt.show()