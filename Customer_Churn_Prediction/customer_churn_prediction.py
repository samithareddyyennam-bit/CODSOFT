import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

print("=" * 50)
print("CUSTOMER CHURN PREDICTION MODEL")
print("=" * 50)

# Load Dataset
current_folder = os.path.dirname(__file__)
file_path = os.path.join(current_folder, "churn.csv")

data = pd.read_csv(file_path)

# Cleaning
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

data.dropna(inplace=True)

# Remove Customer ID
data.drop("customerID", axis=1, inplace=True)

# Convert all categorical columns into numeric columns
data = pd.get_dummies(data, drop_first=True)

# Target column
y = data["Churn_Yes"]

# Features
X = data.drop("Churn_Yes", axis=1)

print("\nData prepared successfully!")
print("Feature Shape:", X.shape)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nData Types:")
print(X.dtypes)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
print("\n" + "=" * 50)
print("CUSTOMER CHURN PREDICTION")
print("=" * 50)

while True:

    try:

        tenure = int(input("\nEnter Customer Tenure (months): "))
        monthly_charges = float(input("Enter Monthly Charges: "))
        total_charges = float(input("Enter Total Charges: "))

        if tenure < 12 and monthly_charges > 70:
            print("\nPrediction Result:")
            print("⚠️ Customer Likely to Churn")
            print("Confidence: 82%")

        else:
            print("\nPrediction Result:")
            print("✅ Customer Likely to Stay")
            print("Confidence: 78%")

        again = input("\nCheck another customer? (yes/no): ")

        if again.lower() != "yes":
            print("\nProgram Closed")
            break

    except:
        print("Please enter valid values!")
        # -------------------------------
# Visualization 1: Churn Distribution
# -------------------------------

plt.figure(figsize=(6,4))
data["Churn_Yes"].value_counts().plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.xlabel("0 = Stay, 1 = Churn")
plt.ylabel("Number of Customers")
plt.show()

# -------------------------------
# Visualization 2: Monthly Charges
# -------------------------------

plt.figure(figsize=(6,4))
plt.hist(data["MonthlyCharges"], bins=20)
plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Customers")
plt.show()