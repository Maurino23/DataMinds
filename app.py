import streamlit as st
import pickle
import numpy as np

# Load the trained Random Forest model
model_path = 'D:\RINO\LOMBA\ICONIC IT 2024\Streamlit\model_rf.pkl'  # Update this path with the correct path to your model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Mapping kelas ke label
class_mapping = {
    0: "Advanced Backend",
    1: "Advanced Data Science",
    2: "Advanced Front End",
    3: "Beginner Backend",
    4: "Beginner Data Science",
    5: "Beginner Front End"
}

# # Set up the Streamlit app layout and style
# st.set_page_config(page_title="DataMinds - Personalized Learning Pathways", layout="centered")
# st.markdown(
#     """
#     <style>
#     .main {
#         background-color: #f0f2f6;
#         font-family: Arial, sans-serif;
#     }
#     header, .reportview-container {
#         background: linear-gradient(to right, #0052cc, #4c6ef5);
#     }
#     header h1, .reportview-container h1 {
#         color: white;
#         font-size: 30px;
#         text-align: center;
#         margin-bottom: 20px;
#     }
#     .stButton button {
#         background-color: #0052cc;
#         color: white;
#         border-radius: 4px;
#         padding: 8px 20px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# App header
st.image("D:\RINO\LOMBA\ICONIC IT 2024\Streamlit\Landscape_DataMind.png", width=800)  # Update the path to your logo image
st.title("DataMinds - Personalized Learning Pathways")

# Input form with two columns
st.subheader("Input Your Data:")
col1, col2 = st.columns(2)

with col1:
    HOURS_DATASCIENCE = st.number_input('Hours Spent on Data Science', min_value=0.0, step=0.5)
    HOURS_BACKEND = st.number_input('Hours Spent on Backend', min_value=0.0, step=0.5)
    HOURS_FRONTEND = st.number_input('Hours Spent on Frontend', min_value=0.0, step=0.5)
    NUM_COURSES_BEGINNER_DATASCIENCE = st.number_input('Number of Beginner Data Science Courses', min_value=0, step=1)
    NUM_COURSES_BEGINNER_BACKEND = st.number_input('Number of Beginner Backend Courses', min_value=0, step=1)
    NUM_COURSES_BEGINNER_FRONTEND = st.number_input('Number of Beginner Frontend Courses', min_value=0, step=1)

with col2:
    NUM_COURSES_ADVANCED_DATASCIENCE = st.number_input('Number of Advanced Data Science Courses', min_value=0, step=1)
    NUM_COURSES_ADVANCED_BACKEND = st.number_input('Number of Advanced Backend Courses', min_value=0, step=1)
    NUM_COURSES_ADVANCED_FRONTEND = st.number_input('Number of Advanced Frontend Courses', min_value=0, step=1)
    AVG_SCORE_DATASCIENCE = st.number_input('Average Score in Data Science', min_value=0.0, max_value=100.0, step=0.1)
    AVG_SCORE_BACKEND = st.number_input('Average Score in Backend', min_value=0.0, max_value=100.0, step=0.1)
    AVG_SCORE_FRONTEND = st.number_input('Average Score in Frontend', min_value=0.0, max_value=100.0, step=0.1)

# Predict button
if st.button('Predict Profile'):
    # Prepare input data for prediction
    input_data = np.array([
        HOURS_DATASCIENCE, HOURS_BACKEND, HOURS_FRONTEND,
        NUM_COURSES_BEGINNER_DATASCIENCE, NUM_COURSES_BEGINNER_BACKEND, NUM_COURSES_BEGINNER_FRONTEND,
        NUM_COURSES_ADVANCED_DATASCIENCE, NUM_COURSES_ADVANCED_BACKEND, NUM_COURSES_ADVANCED_FRONTEND,
        AVG_SCORE_DATASCIENCE, AVG_SCORE_BACKEND, AVG_SCORE_FRONTEND
    ]).reshape(1, -1)

     # Standardize input data using the same scaler used in training
    input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0]

    # Get the predicted profile using manual mapping
    predicted_profile = class_mapping[prediction[0]]

    # Display prediction result
    st.subheader("Recommended Course:")
    # st.success(f"Predicted Profile: **{prediction[0]}**")
    st.success(f"**{predicted_profile}**")
    
    # Sort and display prediction probabilities from highest to lowest
    st.subheader("Courses Suitability:")
    sorted_indices = np.argsort(prediction_proba)[::-1]
    for i in sorted_indices:
        class_name = class_mapping[i]
        probability = prediction_proba[i] * 100
        # Display each class with a progress bar and percentage
        st.write(f"**{class_name}:**")
        st.progress(probability / 100)
        st.write(f"**{probability:.2f}%**")