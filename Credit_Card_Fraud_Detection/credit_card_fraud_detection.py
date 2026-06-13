import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tkinter import *
from tkinter import messagebox

# ----------------------------
# Load Dataset
# ----------------------------
current_folder = os.path.dirname(__file__)
file_path = os.path.join(current_folder, "creditcard.csv")

data = pd.read_csv(file_path)

# Separate legit and fraud
legit = data[data["Class"] == 0]
fraud = data[data["Class"] == 1]

# Balance dataset
legit_sample = legit.sample(n=492, random_state=42)
new_data = pd.concat([legit_sample, fraud], axis=0)

# Features and target
X = new_data.drop(columns=["Class"])
y = new_data["Class"]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Model
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# ----------------------------
# Prediction Function
# ----------------------------
def check_transaction():
    try:
        amount = float(amount_entry.get())
        time = float(time_entry.get())

        # Use a sample transaction
        sample = X.iloc[0].copy()

        # Replace with user input
        sample["Amount"] = amount
        sample["Time"] = time

        sample_df = pd.DataFrame([sample])

        # Scale sample
        sample_scaled = scaler.transform(sample_df)

        # Prediction
        prediction = model.predict(sample_scaled)[0]
        confidence = model.predict_proba(sample_scaled).max() * 100

        if prediction == 1:
            result_label.config(
                text=f"⚠️ Fraudulent Transaction\nConfidence: {confidence:.2f}%",
                fg="red"
            )
        else:
            result_label.config(
                text=f"✅ Legitimate Transaction\nConfidence: {confidence:.2f}%",
                fg="green"
            )

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numbers!"
        )


# ----------------------------
# GUI Window
# ----------------------------
root = Tk()
root.title("Credit Card Fraud Detection")
root.geometry("500x450")

title = Label(
    root,
    text="Credit Card Fraud Detection",
    font=("Arial", 18, "bold")
)
title.pack(pady=15)

accuracy_label = Label(
    root,
    text=f"Model Accuracy: {accuracy * 100:.2f}%",
    font=("Arial", 11)
)
accuracy_label.pack()

# Amount
Label(root, text="Enter Amount:", font=("Arial", 12)).pack(pady=5)

amount_entry = Entry(root, width=30, font=("Arial", 12))
amount_entry.pack()

# Time
Label(root, text="Enter Time:", font=("Arial", 12)).pack(pady=5)

time_entry = Entry(root, width=30, font=("Arial", 12))
time_entry.pack()

# Button
Button(
    root,
    text="Check Transaction",
    font=("Arial", 12, "bold"),
    command=check_transaction
).pack(pady=20)

# Result
result_label = Label(
    root,
    text="Result will appear here",
    font=("Arial", 14, "bold")
)
result_label.pack(pady=20)

root.mainloop()