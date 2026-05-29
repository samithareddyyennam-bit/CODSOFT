Spam SMS Detection using Machine Learning

Project Overview

This project is a Machine Learning based Spam SMS Detection System developed using Python. The model predicts whether an SMS message is Spam or Not Spam (Ham).

The project uses Natural Language Processing (NLP) techniques and a Naive Bayes Classifier to analyze text messages and classify them accurately.

Features

- Spam and Non-Spam message detection
- User input message prediction
- Prediction confidence percentage
- Confusion Matrix visualization
- Accuracy calculation of the model

Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- NLP (TF-IDF Vectorization)

Machine Learning Algorithm

Multinomial Naive Bayes

Dataset

The dataset used is the SMS Spam Collection Dataset, containing SMS messages labeled as spam or ham.

Project Workflow

1. Load Dataset
2. Clean Data
3. Convert Text into Numerical Form using TF-IDF
4. Train Machine Learning Model
5. Test Accuracy
6. Predict User Input Messages

Accuracy

The model achieved approximately 96%–97% accuracy.

How to Run the Project

1. Install required libraries:
   pip install pandas scikit-learn matplotlib nltk

2. Run the Python file:
   python spam_sms_detection.py

Example

Input:
"Congratulations! You won ₹5000"

Output:
SPAM