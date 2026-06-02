import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


print("=" * 50)
print("MOVIE GENRE CLASSIFICATION MODEL")
print("=" * 50)

# Get current folder path
current_folder = os.path.dirname(__file__)

# Dataset path
file_path = os.path.join(current_folder, "train_data.txt")

# Load dataset
data = pd.read_csv(
    file_path,
    sep=":::",
    names=["id", "title", "genre", "description"],
    engine="python"
)

# Remove missing values
data.dropna(inplace=True)

# Smaller dataset for faster execution
data = data.sample(8000, random_state=42)

# Features and target
X = data["description"]
y = data["genre"]

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=300)

model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Predict test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Continuous prediction
while True:
    movie_description = input(
        "\nEnter movie description (or type 'exit' to stop): "
    )

    # Exit program
    if movie_description.lower() == "exit":
        print("\nProgram Closed")
        break

    # Convert text into numbers
    movie_vector = vectorizer.transform(
        [movie_description]
    )

    # Predict genre
    prediction = model.predict(movie_vector)

    # Confidence score
    probability = model.predict_proba(
        movie_vector
    )

    print("\nPrediction Result:")
    print(
        "Predicted Genre:",
        prediction[0].strip()
    )

    print(
        f"Confidence: {max(probability[0]) * 100:.2f}%"
    )

# Graph after closing program
genre_counts = (
    data["genre"]
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10, 5))
genre_counts.plot(kind="bar")

plt.title("Top 10 Movie Genres")
plt.xlabel("Genre")
plt.ylabel("Count")

plt.show()