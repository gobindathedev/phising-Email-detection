import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load dataset
data = pd.read_csv("C:/Users/gdas7/OneDrive/Desktop/phising email detection/spam.csv")

# Clean data
data.drop_duplicates(inplace=True)

# Make sure columns are correctly named
data.rename(columns={'Category': 'category', 'Message': 'message'}, inplace=True)
data['category'] = data['category'].replace(['ham', 'spam'], ['Not spam', 'Spam'])

# Features & labels
mess = data['message']
cat = data['category']

# Train-test split
mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2, random_state=42)

# Vectorization
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

# Model
model = MultinomialNB()
model.fit(features, cat_train)

# Accuracy
accuracy = model.score(cv.transform(mess_test), cat_test)

# Prediction function
def predict(message):
    input_message = cv.transform([message])
    return model.predict(input_message)[0]

# Streamlit UI
st.title("📧 Phishing Email / Spam Detection with Machine Learning")
st.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")

user_input = st.text_area("Enter the email/message text:")

if st.button("Predict"):
    prediction = predict(user_input)
    if prediction == "Spam":
        st.error("🚨 This message is SPAM / PHISHING!")
    else:
        st.success("✅ This message is SAFE (Not spam).")
