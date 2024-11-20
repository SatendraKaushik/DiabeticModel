import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
  .main{
            margin:0px
            padding:0px;
            }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
            color:black
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        color:black
    }
    .metric-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color:black;
    }
    .metric-label {
        font-size: 1em;
        color: #666;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #FF4B4B;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    with open('diabetes_model.pkl', 'rb') as model_file:
        classifier = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return classifier, scaler

classifier, scaler = load_model()

# Header section
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://img.icons8.com/color/96/000000/diabetes.png", width=80)
with col2:
    st.title("Diabetes Risk Prediction System")
    st.markdown("<p style='color: #666; font-size: 1.2em;'>An AI-powered tool for early diabetes risk assessment</p>", 
                unsafe_allow_html=True)

# Information box
st.markdown("""
    <div class='info-box'>
        <h4>👋 Welcome to the Diabetes Risk Predictor</h4>
        <p>This tool uses machine learning to assess diabetes risk based on clinical parameters. 
        Please fill in all the fields below with accurate information for the best results.</p>
    </div>
    """, unsafe_allow_html=True)

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Personal Information")
    age = st.number_input('Age (years)', 
                         min_value=20, 
                         max_value=100, 
                         value=31,
                         help="Enter age between 20 and 100 years")
    
    pregnancies = st.number_input('Number of Pregnancies',
                                min_value=0,
                                max_value=20,
                                value=1,
                                help="Enter number of pregnancies (0-20)")
    
    bmi = st.number_input('BMI (kg/m²)',
                         min_value=10.0,
                         max_value=50.0,
                         value=26.6,
                         format="%.1f",
                         help="Body Mass Index (weight in kg/(height in m)²)")
    
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function',
                                               min_value=0.0,
                                               max_value=3.0,
                                               value=0.351,
                                               format="%.3f",
                                               help="Diabetes mellitus history in relatives")

with col2:
    st.markdown("### 🔬 Clinical Measurements")
    glucose = st.number_input('Glucose Level (mg/dL)',
                            min_value=0,
                            max_value=200,
                            value=85,
                            help="Fasting blood glucose level")
    
    blood_pressure = st.number_input('Blood Pressure (mm Hg)',
                                   min_value=0,
                                   max_value=200,
                                   value=66,
                                   help="Diastolic blood pressure")
    
    skin_thickness = st.number_input('Skin Thickness (mm)',
                                   min_value=0,
                                   max_value=100,
                                   value=29,
                                   help="Triceps skin fold thickness")
    
    insulin = st.number_input('Insulin Level (mu U/ml)',
                            min_value=0,
                            max_value=1000,
                            value=0,
                            help="2-Hour serum insulin")

# Create a spacer
st.markdown("<br>", unsafe_allow_html=True)

# Prediction button with custom styling
predict_btn = st.button('Analyze Risk')

if predict_btn:
    # Show a spinner while processing
    with st.spinner('Analyzing data...'):
        # Prepare input data
        input_data = {
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree_function],
            'Age': [age]
        }
        
        # Convert to DataFrame and make prediction
        input_data_df = pd.DataFrame(input_data)
        std_data = scaler.transform(input_data_df)
        prediction = classifier.predict(std_data)

        # Create columns for the result display
        col1, col2, col3 = st.columns([1,2,1])
        
        with col2:
            if prediction[0] == 1:
                st.markdown("""
                    <div class='prediction-box' style='background-color: #FFE5E5;'>
                        <h2 style='color: #FF4B4B;'>⚠️ High Risk Detected</h2>
                        <p>The analysis indicates potential diabetes risk.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='prediction-box' style='background-color: #E5FFE5;'>
                        <h2 style='color: #28A745;'>✅ Low Risk Detected</h2>
                        <p>The analysis indicates low diabetes risk.</p>
                    </div>
                    """, unsafe_allow_html=True)

        # Display key metrics
        st.markdown("### 📊 Key Risk Factors")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown("""
                <div class='metric-container'>
                    <div class='metric-label'>Glucose Level</div>
                    <div class='metric-value'>{} mg/dL</div>
                </div>
            """.format(glucose), unsafe_allow_html=True)
            
        with metric_col2:
            st.markdown("""
                <div class='metric-container'>
                    <div class='metric-label'>BMI</div>
                    <div class='metric-value'>{:.1f}</div>
                </div>
            """.format(bmi), unsafe_allow_html=True)
            
        with metric_col3:
            st.markdown("""
                <div class='metric-container'>
                    <div class='metric-label'>Blood Pressure</div>
                    <div class='metric-value'>{} mm Hg</div>
                </div>
            """.format(blood_pressure), unsafe_allow_html=True)

        # Recommendations section
        st.markdown("### 💡 Recommendations")
        st.markdown("""
            <div class='info-box'>
                <h4>Next Steps:</h4>
                <ul>
                    <li>Regular monitoring of blood glucose levels</li>
                    <li>Maintain a healthy diet and exercise routine</li>
                    <li>Schedule regular check-ups with healthcare providers</li>
                    <li>Keep track of any changes in symptoms or health conditions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)



