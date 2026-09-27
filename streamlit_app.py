
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="EduGuard AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 EduGuard AI")
st.subheader("Student Performance Prediction and Early Warning System")
st.write(
    "Enter a student's information and early learning activity "
    "to receive a predicted academic outcome."
)

st.info(
    "This is an experimental model. Its prediction is an estimate, "
    "not a guaranteed student outcome."
)

with st.form("student_prediction_form"):

    st.header("1. Course Information")

    col1, col2 = st.columns(2)

    with col1:
        code_module = st.selectbox(
            "Module",
            ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
        )

    with col2:
        code_presentation = st.selectbox(
            "Presentation",
            ["2013J", "2014J", "2013B", "2014B"]
        )

    date_registration_num = st.number_input(
        "Registration date (days relative to course start)",
        value=-30.0,
        step=1.0
    )

    st.header("2. Early Learning Activity")

    col1, col2, col3 = st.columns(3)

    with col1:
        early_assessment_count = st.number_input(
            "Early assessments completed",
            min_value=0,
            value=2,
            step=1
        )

        early_mean_score = st.number_input(
            "Average early assessment score",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        early_max_score = st.number_input(
            "Highest early assessment score",
            min_value=0.0,
            max_value=100.0,
            value=85.0
        )

        early_min_score = st.number_input(
            "Lowest early assessment score",
            min_value=0.0,
            max_value=100.0,
            value=65.0
        )

    with col2:
        early_total_clicks = st.number_input(
            "Total VLE clicks by Day 28",
            min_value=0,
            value=120,
            step=1
        )

        early_active_days = st.number_input(
            "Active learning days",
            min_value=0,
            max_value=28,
            value=15,
            step=1
        )

        early_unique_sites = st.number_input(
            "Unique learning resources accessed",
            min_value=0,
            value=8,
            step=1
        )

    st.header("3. Student Background")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["M", "F"])

        region = st.selectbox(
            "Region",
            [
                "East Anglian Region",
                "Scotland",
                "North Western Region",
                "South East Region",
                "West Midlands Region",
                "Wales",
                "North Region",
                "South Region",
                "South West Region",
                "East Midlands Region",
                "Yorkshire Region",
                "London Region",
                "South West England",
                "East Midlands",
                "Other Region"
            ]
        )

        highest_education = st.selectbox(
            "Highest education",
            [
                "A Level or Equivalent",
                "HE Qualification",
                "Lower Than A Level",
                "Post Graduate Qualification",
                "No Formal quals"
            ]
        )

    with col2:
        imd_band = st.selectbox(
            "IMD band",
            [
                "0-10%",
                "10-20%",
                "20-30%",
                "30-40%",
                "40-50%",
                "50-60%",
                "60-70%",
                "70-80%",
                "80-90%",
                "90-100%",
                "Unknown"
            ]
        )

        age_band = st.selectbox(
            "Age band",
            ["0-35", "35-55", "55<="]
        )

        num_of_prev_attempts = st.number_input(
            "Previous attempts",
            min_value=0,
            value=0,
            step=1
        )

    with col3:
        studied_credits = st.number_input(
            "Studied credits",
            min_value=0,
            value=60,
            step=1
        )

        disability = st.selectbox(
            "Declared disability",
            ["N", "Y"]
        )

    submitted = st.form_submit_button(
        "Predict Student Outcome",
        type="primary",
        use_container_width=True
    )

if submitted:

    student_data = {
        "code_module": code_module,
        "code_presentation": code_presentation,
        "date_registration_num": date_registration_num,
        "early_assessment_count": early_assessment_count,
        "early_mean_score": early_mean_score,
        "early_max_score": early_max_score,
        "early_min_score": early_min_score,
        "early_total_clicks": early_total_clicks,
        "early_active_days": early_active_days,
        "early_unique_sites": early_unique_sites,
        "gender": gender,
        "region": region,
        "highest_education": highest_education,
        "imd_band": imd_band,
        "age_band": age_band,
        "num_of_prev_attempts": num_of_prev_attempts,
        "studied_credits": studied_credits,
        "disability": disability
    }

    try:
        with st.spinner("Getting prediction from EduGuard AI..."):
            response = requests.post(
                API_URL,
                json=student_data,
                timeout=30
            )

        if response.status_code == 200:
            result = response.json()

            st.header("Prediction Result")

            outcome = result["predicted_outcome"]
            risk = result["risk_probability"]

            st.metric("Predicted Outcome", outcome)
            st.metric("Estimated Risk of Fail or Withdrawal", f"{risk:.1%}")

            st.subheader("Model-estimated class probabilities")

            probabilities = result["class_probabilities"]

            for class_name, probability in probabilities.items():
                st.write(f"**{class_name}: {probability:.1%}**")
                st.progress(float(probability))

            st.caption(result.get("note", ""))

            st.warning(
                "Use this result as an early-support signal, not as a "
                "final judgment about a student. The model's probabilities "
                "are estimates and may be inaccurate."
            )

        else:
            st.error(
                f"API returned status {response.status_code}: "
                f"{response.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the prediction API. "
            "Make sure the FastAPI server is running at "
            "http://127.0.0.1:8000."
        )

    except requests.exceptions.RequestException as error:
        st.error(f"Request failed: {error}")
