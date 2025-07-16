import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

st.title("🧠 Text Classifier: Decision Tree & Random Forest")

st.markdown("""
Sube un archivo CSV que contenga al menos dos columnas:
- Una columna con el texto (ejemplo: `texto`)
- Una columna con las etiquetas (ejemplo: `etiqueta`)
""")

# Subir archivo
uploaded_file = st.file_uploader("📁 Sube tu archivo CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Vista previa de los datos")
    st.write(df.head())

    # Selección de columnas
    text_column = st.selectbox("📝 Selecciona la columna de texto", df.columns)
    label_column = st.selectbox("🏷️ Selecciona la columna de etiquetas", df.columns)

    X = df[text_column].astype(str)
    y = df[label_column].astype(str)

    vectorizer = TfidfVectorizer()
    X_vect = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

    classifier_name = st.selectbox("🔍 Selecciona un modelo", ["Decision Tree", "Random Forest"])

    if st.button("🚀 Entrenar modelo"):
        if classifier_name == "Decision Tree":
            clf = DecisionTreeClassifier(random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=100, random_state=42)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.subheader("📈 Resultados del modelo")
        st.write(f"**Precisión:** {accuracy_score(y_test, y_pred):.2f}")
        st.text("Reporte de clasificación:\n" + classification_report(y_test, y_pred))

        st.subheader("🧪 Clasifica tu propio texto")
        user_input = st.text_area("Introduce un texto para clasificar")

        if st.button("Predecir"):
            input_vect = vectorizer.transform([user_input])
            prediction = clf.predict(input_vect)
            st.success(f"Etiqueta predicha: **{prediction[0]}**")
