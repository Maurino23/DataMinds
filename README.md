# DataMinds: Predictive Learning Pathways in Education

Welcome to the repository of **DataMinds**, a collaborative project by students from Universitas Islam Bandung and Politeknik Astra for ICONIC IT 2024 at Universitas Siliwangi. This project is aimed at revolutionizing the educational landscape by leveraging data science and machine learning to optimize students' learning pathways.

## 🌟 Overview
In the digital age, adapting educational methods to technological advancements is critical. This project seeks to build a predictive model using **Random Forest** to identify the optimal learning path for students based on their academic profile and engagement. Additionally, an interactive web application built with **Streamlit** enables users to input academic data and receive tailored learning recommendations.

## ✨ Key Features
- **Data Analysis**: Comprehensive exploratory data analysis (EDA) to understand patterns and trends.
- **Model Building**: A well-optimized Random Forest model using `GridSearchCV` for hyperparameter tuning.
- **Interactive Web App**: A user-friendly interface that provides predictions and probability scores.
- **Deployment**: The application is deployed for public access through **Streamlit Sharing**.

## 🔍 Problem Statement
The rapid advancement of technology requires educational institutions to personalize learning experiences to meet diverse needs. Identifying key factors that influence a student's learning pathway can enhance educational outcomes and minimize incorrect course selections.

## 🎯 Objectives
1. **Analysis**: Identify factors influencing learning pathways.
2. **Prediction**: Build a robust model to predict the most suitable learning path.
3. **Accessibility**: Deploy a web application for intuitive data input and output display.

## 📂 Dataset
The dataset, sourced from ICONIC IT 2024, encompasses:
- **Independent Variables**: Hours spent in learning data science, backend, frontend, course counts (beginner and advanced levels), and average scores.
- **Dependent Variable**: Learning profile (`PROFILE`), categorized into different proficiency levels.

## 🔧 Tools & Technologies
- **Python**: Programming language for data analysis and model development.
- **Scikit-learn**: For model building and hyperparameter tuning.
- **Pandas & NumPy**: Data manipulation and preprocessing.
- **Matplotlib & Seaborn**: Visualizations for EDA.
- **Streamlit**: For building the web interface.
- **Jupyter Notebook**: For prototyping and initial analysis.

## 📈 Methodology
1. **Data Preprocessing**:
   - Removal of irrelevant columns (e.g., `NAME`, `USER_ID`).
   - Handling missing values using median imputation.
   - Encoding categorical data and normalizing numerical data.
2. **Model Training**:
   - Initial model setup with **Random Forest**.
   - Hyperparameter tuning using **GridSearchCV**.
3. **Evaluation**:
   - Assessing model performance using accuracy, precision, recall, and F1-score.
4. **Web Application**:
   - Building an interactive app for user input and prediction display.
   - Deploying via **Streamlit Sharing** for public access.

## 🚀 Results
- The model achieved an impressive accuracy of **91%**, with balanced precision, recall, and F1-score.
- Key features identified were `HOURS_BACKEND` and `HOURS_DATASCIENCE`, highlighting their strong influence on student learning profiles.
## 🌐 Access the Application
Try our web app: [DataMinds Learning Pathways App](https://dataminds-pathways.streamlit.app/)

## 📚 References & Resources
- [GitHub Repository](https://github.com/Maurino23/DataMinds)
- [Presentation Video](https://drive.google.com/file/d/1C5KPkSDLmMSaPMACYCA53zn3YclkZdbN/view?usp=sharing)
- [Google Colab Notebook](https://colab.research.google.com/drive/163YXgx4BJVYVXnij33TIsvOdWxY7uryv?usp=sharing)

## 👏 Acknowledgements
We thank all stakeholders, mentors, and peers who provided support throughout the development of this project.
