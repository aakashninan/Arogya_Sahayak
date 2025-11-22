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
from io import StringIO 

# --- Configuration ---
TARGET_COL = 'charges'
SEED = 42
# Placeholder conversion rate: 1 USD ≈ 83 INR (Using a typical rate)
USD_TO_INR_RATE = 83.0 
APP_TITLE = "Aarogya Sahayak - Health Cost Estimator"

# Define which columns are numerical and which are categorical
NUMERICAL_FEATURES = ['age', 'bmi', 'children']
CATEGORICAL_FEATURES = ['sex', 'smoker', 'region']
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# --- EMBEDDED DATA (This guarantees the app loads) ---
# Content of insurance.csv copied into a Python multi-line string
EMBEDDED_CSV_DATA = """age,sex,bmi,children,smoker,region,charges
19,female,27.9,0,yes,southwest,16884.924
18,male,33.77,1,no,southeast,1725.5523
28,male,33,3,no,southeast,4449.462
33,male,22.705,0,no,northwest,21984.47061
32,male,28.88,0,no,northwest,3866.8552
31,female,25.74,0,no,southeast,3756.6216
46,female,33.44,1,no,southeast,8240.5896
37,female,27.74,3,no,northwest,7281.5056
37,male,29.83,2,no,northeast,6406.4107
60,female,25.84,0,no,northwest,28923.13692
25,male,26.22,0,no,northeast,2721.3208
62,female,26.29,0,yes,southeast,27808.7251
23,male,34.4,0,no,southwest,1826.843
56,female,39.82,0,no,southeast,11090.7178
27,male,42.13,0,yes,southeast,39611.7577
19,male,24.6,1,no,southwest,1837.28
52,female,30.78,1,no,northeast,10797.3172
23,male,23.84,0,no,northeast,2395.17
56,male,40.3,0,no,southwest,10602.385
30,male,35.3,0,yes,southwest,36837.467
60,female,36.005,0,no,northeast,13228.8468
30,female,32.4,1,no,southwest,4149.683
18,male,34.1,0,no,southeast,1137.011
34,female,31.92,1,yes,northeast,37701.8768
37,male,28.025,2,no,northwest,6203.93
43,male,37.8,0,no,southwest,4211.849
55,female,35.2,0,no,southeast,13495.435
63,female,33.11,0,no,northwest,14474.38394
31,female,36.63,2,no,southeast,3873.8017
22,male,35.6,0,yes,southwest,35585.576
31,male,36.3,2,yes,southwest,38711.0
22,male,35.6,0,yes,southwest,35585.576
31,female,25.74,0,no,southeast,3756.6216
40,male,33.33,0,no,southeast,9298.0807
28,female,34.77,0,no,northwest,3556.9223
"""

# ===========================================================
# 🧩 Data Loading and Model Training Pipeline
# ===========================================================

@st.cache_resource
def load_and_train_model():
    """Loads embedded data, defines preprocessing steps, and trains the full pipeline."""
    
    # --- 1. Prepare Data ---
    df = pd.read_csv(StringIO(EMBEDDED_CSV_DATA))
    
    # Ensure data types are correct after reading from string
    df['age'] = df['age'].astype(int)
    df['children'] = df['children'].astype(int)
    df['bmi'] = df['bmi'].astype(float)
    df['charges'] = df['charges'].astype(float)
    
    X = df.drop(TARGET_COL, axis=1)
    
    # --- CRITICAL FIX 1: Log Transform Target ---
    # Apply log transformation to the target variable (y) to handle skewness.
    y = np.log(df[TARGET_COL])
    
    # --- 2. Define Preprocessing ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
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
    # Train model on the LOG-TRANSFORMED Y
    model_pipeline.fit(X_train, y_train)

    # --- 5. Evaluate and Get Metrics ---
    y_pred_log = model_pipeline.predict(X_test)
    
    # Metrics must be calculated on the REVERSE-TRANSFORMED Y
    y_test_original = np.exp(y_test)
    y_pred_original = np.exp(y_pred_log)
    
    r2 = r2_score(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    
    return model_pipeline, r2, rmse

# Load and train the model/pipeline
model_pipeline, r2_score_val, rmse_val = load_and_train_model()

# ===========================================================
# ⚙️ Prediction Function
# ===========================================================

def predict_cost(input_df):
    """Uses the trained pipeline to predict the cost (log-transformed) 
       and returns the cost in original scale."""
       
    # Predict the log(cost)
    predicted_log_cost = model_pipeline.predict(input_df)
    
    # --- CRITICAL FIX 2: Reverse Transform Prediction ---
    # Reverse the transformation (exponentiate) to get the final cost
    predicted_cost = np.exp(predicted_log_cost[0])
    return predicted_cost

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
    
    input_container = st.container(border=True)
    
    with input_container:
        st.subheader("1. Your Personal Data")
        
        col_age, col_sex, col_children = st.columns(3)
        
        with col_age:
            # Age - NUMBER INPUT
            age = st.number_input('Age (Years)', min_value=18, max_value=64, value=39, step=1,
                                  help="Your age is the number one non-lifestyle risk factor.")
        
        with col_sex:
            # Sex - Clear label
            sex = st.radio('Gender', options=['male', 'female'], horizontal=True)
            
        with col_children:
            # Children - NUMBER INPUT
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


    # Create DataFrame from user inputs 
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
        
        # Predict the cost (already reverse-transformed inside predict_cost function)
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
    st.markdown("This model uses **Linear Regression** to predict a numerical cost, relying on a clean, scalable data pipeline. **The target variable is log-transformed to handle data skewness, which provides more realistic estimates.**")
    
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
# 1. LOG TRANSFORMATION applied to 'charges' (y) before training the model.
# 2. OneHotEncoder for categorical features (sex, smoker, region).

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['sex', 'smoker', 'region'])
    ],
    remainder='passthrough' 
)

# Full Pipeline: Preprocessing -> Linear Regression
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
# 3. REVERSE TRANSFORMATION (exponentiation) applied after prediction.
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
