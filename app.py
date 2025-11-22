import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

# --- Configuration ---
DATA_FILE = 'insurance.csv'
TARGET_COL = 'charges'
SEED = 42
# Placeholder conversion rate: 1 USD ≈ 83 INR (Using a typical rate)
USD_TO_INR_RATE = 83.0 
APP_TITLE = "Aarogya Sahayak - Health Cost Estimator"

# Define which columns are numerical and which are categorical
NUMERICAL_FEATURES = ['age', 'bmi', 'children']
CATEGORICAL_FEATURES = ['sex', 'smoker', 'region']
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ===========================================================
# 🧩 Data Loading and Model Training Pipeline
# ===========================================================

@st.cache_resource
def load_and_train_model(data_path):
    """Loads data, defines preprocessing steps, and trains the full pipeline."""
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{DATA_FILE}' was not found. Please ensure it is in the same directory.")
        st.stop()
        
    # --- 1. Prepare Data ---
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    
    # --- 2. Define Preprocessing ---
    # OneHotEncoder handles the conversion of categorical text to numbers
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CATEGORICAL_FEATURES)
        ],
        remainder='passthrough' # Keep numerical columns (age, bmi, children) as they are
    )

    # --- 3. Define the Full Pipeline ---
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # --- 4. Train/Fit the Pipeline ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    model_pipeline.fit(X_train, y_train)

    # --- 5. Evaluate and Get Metrics ---
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model_pipeline, r2, rmse

# Load and train the model/pipeline
model_pipeline, r2_score_val, rmse_val = load_and_train_model(DATA_FILE)

# ===========================================================
# ⚙️ Prediction Function
# ===========================================================

def predict_cost(input_df):
    """Uses the trained pipeline to predict the cost."""
    prediction = model_pipeline.predict(input_df)
    return prediction[0]

# ===========================================================
# ⚛️ Streamlit Interface Code
# ===========================================================

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"🏥 {APP_TITLE}")
st.markdown("---") # Visual separator

# --- Navigation Tabs ---
tab1, tab2 = st.tabs(["📊 Cost Calculator", "🧠 Model Details"])

with tab1:
    st.header("Patient Profile Builder")
    st.markdown("#### Enter your health and demographic details to estimate the annual health cost in Rupees.")

    # --- User Input Collection (Optimized for Readability) ---
    
    # Use st.container for a visually separated input area
    input_container = st.container(border=True)
    
    with input_container:
        st.subheader("1. Your Personal Data")
        
        col_age, col_sex, col_children = st.columns(3)
        
        with col_age:
            # Age - CHANGED TO NUMBER INPUT
            age = st.number_input('Age (Years)', min_value=18, max_value=64, value=39, step=1,
                                  help="Your age is the number one non-lifestyle risk factor.")
        
        with col_sex:
            # Sex - Clear label
            sex = st.radio('Gender', options=['male', 'female'], horizontal=True)
            
        with col_children:
            # Children - CHANGED TO NUMBER INPUT
            children = st.number_input('Dependents/Children', min_value=0, max_value=5, value=1, step=1,
                                       help="Number of people covered under your policy.")

        st.markdown("---")
        st.subheader("2. Key Risk Factors")

        col_bmi, col_smoker, col_region = st.columns(3)
        
        with col_bmi:
            # BMI - Large number input with context
            bmi = st.number_input('Body Mass Index (BMI)', 
                                  min_value=15.0, max_value=55.0, value=30.0, step=0.5,
                                  help="BMI is calculated from your weight and height. Typical healthy range is 18.5 to 25.")

        with col_smoker:
            # Smoker - Clear, direct Yes/No question
            smoker = st.radio('Do you currently smoke?', options=['yes', 'no'], horizontal=True)
            
        with col_region:
            # Region - Clear geographic selector
            region = st.selectbox('Residential Region (US Data)', 
                                  options=['northeast', 'northwest', 'southeast', 'southwest'],
                                  help="Geographic location used in the training data.")


    # Create DataFrame from user inputs (Removed redundant st.dataframe display)
    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }], columns=ALL_FEATURES)

    st.markdown("---")

    # --- Prediction Button and Output ---
    if st.button('💰 ESTIMATE MY ANNUAL COST IN RUPEES', type="primary", use_container_width=True):
        
        # Predict the cost
        predicted_cost_usd = predict_cost(input_data)
        
        # Convert to INR
        predicted_cost_inr = predicted_cost_usd * USD_TO_INR_RATE
        
        # Display the result
        st.success("✅ ESTIMATION COMPLETE")
        
        # Use a container for the large result display
        result_container = st.container(border=True)
        
        with result_container:
            st.markdown("### Your Estimated Annual Health Charge:")
            
            # Display INR Result prominently and clearly (only INR)
            st.markdown(f"""
            # **₹{predicted_cost_inr:,.0f}**
            """)
            
            st.markdown(f"*(Based on an approximate conversion rate of 1 USD = {USD_TO_INR_RATE:.2f} INR)*")

            # --- Interpretation ---
            st.markdown("---")
            st.subheader("💡 Key Factors Influencing This Estimate")
            
            # Interpretation based on key feature (Smoker is the biggest predictor in this dataset)
            if smoker == 'yes':
                st.warning("⚠️ **High Risk:** Your smoking status is the single largest factor driving up this estimated cost.")
            else:
                 st.info("✅ **Low Risk:** Your non-smoking status helps keep the estimate lower.")
            
            st.info("📈 **Age and BMI:** Costs generally increase with age, and a high BMI contributes to higher estimates due to related health risks.")
            st.info("👶 **Dependents:** The number of children/dependents slightly influences the estimated premium.")

with tab2:
    st.header("Model Technical Details")
    st.markdown("This model uses **Linear Regression** to predict a numerical cost, relying on a clean, scalable data pipeline.")
    
    st.subheader("Evaluation Metrics (Test Set)")
    col_r2, col_rmse = st.columns(2)
    
    with col_r2:
        st.metric(label="R² Score (Model Fit)", value=f"{r2_score_val:.4f}", help="Closer to 1.0 means the model fits the data better.")
    with col_rmse:
        # Display RMSE in INR for context
        st.metric(label="RMSE (Typical Error)", value=f"₹{(rmse_val * USD_TO_INR_RATE):,.0f}", help="This is the average error margin (in Rupees) for a prediction.")

    st.subheader("Data Preprocessing Pipeline")
    st.code("""
# The pipeline ensures consistency:
preprocessor = ColumnTransformer(
    transformers=[
        # Converts 'male', 'female', 'yes', 'no', 'region' names into numerical 0s and 1s
        ('cat', OneHotEncoder(drop='first'), ['sex', 'smoker', 'region'])
    ],
    remainder='passthrough' # Keeps 'age', 'bmi', 'children' columns raw
)

# Full Pipeline: Preprocessing -> Linear Regression
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
    """)

# --- Copyright Footer ---
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    color: #808080;
    text-align: center;
    padding: 5px;
    font-size: 10pt;
    font-family: Arial, sans-serif;
    border-top: 1px solid #ddd;
}
</style>
<div class="footer">
    © 2025 Aarogya Sahayak. All rights reserved. | For educational demonstration only.
</div>
""", unsafe_allow_html=True)
