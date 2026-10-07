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
    "Введіть дані студента, щоб спрогнозувати "
    "його підсумковий результат."
)

attendance = st.slider(
    "Відвідуваність (%)",
    min_value=0,
    max_value=100,
    value=80,
    step=1
)

midterm = st.number_input(
    "Оцінка за проміжний контроль",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)

final = st.number_input(
    "Оцінка за фінальний іспит",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

assignments_avg = st.number_input(
    "Середня оцінка за завдання",
    min_value=0.0,
    max_value=100.0,
    value=80.0,
    step=1.0
)

quizzes_avg = st.number_input(
    "Середня оцінка за тести",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

study_hours = st.number_input(
    "Кількість годин навчання на тиждень",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=1.0
)

def preprocess_input(
    attendance,
    midterm,
    final,
    assignments_avg,
    quizzes_avg,
    study_hours,
    preprocessor
):
    df = pd.DataFrame({
        "Attendance": [attendance],
        "Midterm": [midterm],
        "Final": [final],
        "Assignments_Avg": [assignments_avg],
        "Quizzes_Avg": [quizzes_avg],
        "Study_Hours_per_Week": [study_hours]
    })

    df_processed = preprocessor.transform(df)

    return df_processed

if st.button("Спрогнозувати результат"):

    input_data = preprocess_input(
        attendance,
        midterm,
        final,
        assignments_avg,
        quizzes_avg,
        study_hours,
        preprocessor
    )

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
