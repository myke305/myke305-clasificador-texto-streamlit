import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.title("Text Classifier: Decision Tree & Random Forest")

st.write("""
Upload a CSV file containing at least two columns:
- One column with the text (e.g., 'text')
- One column with the labels (e.g., 'label')
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    text_column = st.selectbox("Select the text column", df.columns)
    label_column = st.selectbox("Select the label column", df.columns)

    X = df[text_column].astype(str)
    y = df[label_column].astype(str)

    vectorizer = TfidfVectorizer()
    X_vect = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

    classifier_name = st.selectbox("Select classifier", ["Decision Tree", "Random Forest"])

    if st.button("Train Classifier"):
        if classifier_name == "Decision Tree":
            clf = DecisionTreeClassifier(random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=100, random_state=42)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.subheader("Results")
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:\n" + classification_report(y_test, y_pred))

        st.subheader("Try with your own text")
        user_text = st.text_area("Enter text to classify")
        if st.button("Predict"):
            user_vect = vectorizer.transform([user_text])
            prediction = clf.predict(user_vect)
            st.write(f"**Predicted label:** {prediction[0]}")