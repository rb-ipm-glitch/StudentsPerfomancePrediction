import joblib
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

@st.cache_resource
def load_model_and_resources():
    model = load_model(
        "student_performance_model.keras",
        compile=False
    )

    preprocessor = joblib.load(
        "preprocessor.pkl"
    )

    return model, preprocessor


model, preprocessor = load_model_and_resources()

st.title("🎓 Прогнозування успішності студента")

st.write(
    "Введіть дані студента для прогнозування "
    "його підсумкового результату."
)

col1, col2 = st.columns(2)

with col1:

    department = st.selectbox(
        "Факультет",
        [
            "CS",
            "Engineering",
            "Business",
            "Mathematics"
        ]
    )

    attendance = st.slider(
        "Відвідуваність (%)",
        min_value=0,
        max_value=100,
        value=80
    )

    midterm = st.number_input(
        "Оцінка за проміжний контроль",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    final = st.number_input(
        "Оцінка за фінальний іспит",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )


with col2:

    assignments_avg = st.number_input(
        "Середня оцінка за завдання",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

    quizzes_avg = st.number_input(
        "Середня оцінка за тести",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )

    study_hours = st.number_input(
        "Годин навчання на тиждень",
        min_value=0.0,
        max_value=100.0,
        value=10.0
    )
    
def preprocess_input():

    data = pd.DataFrame({
        "Department": [department],
        "Attendance (%)": [attendance],
        "Midterm_Score": [midterm],
        "Final_Score": [final],
        "Assignments_Avg": [assignments_avg],
        "Quizzes_Avg": [quizzes_avg],
        "Study_Hours_per_Week": [study_hours]
    })

    return preprocessor.transform(data)

if st.button(
    "Спрогнозувати результат",
    type="primary",
    use_container_width=True
):

    input_data = preprocess_input()

    prediction = model.predict(
        input_data,
        verbose=0
    )

    predicted_score = float(
        prediction[0][0]
    )

    st.success(
        f"Прогнозований результат студента: "
        f"{predicted_score:.2f}"
    )
