import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for a clean UI
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-result {
    font-size: 1.25rem;
    font-weight: bold;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin-top: 1.5rem;
}
.positive-prediction {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}
.negative-prediction {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}
</style>
""", unsafe_allow_html=True)

# Load models and components once
@st.cache_resource
def load_components():
    """Load the trained model, PCA transformer, scaler, and feature list."""
    try:
        model = joblib.load('best_model_xgboost.pkl')
        pca = joblib.load('pca_transformer.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Load feature names and means from the new JSON files
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        with open('column_means.json', 'r') as f:
            column_means = json.load(f)

        # Get the list of continuous features the scaler was trained on
        continuous_features = scaler.feature_names_in_
        
        return model, pca, scaler, feature_names, column_means, continuous_features
    except FileNotFoundError as e:
        st.error(f"Error loading a required file: {e}. Please ensure all model files (best_model_decision_tree.pkl, pca_transformer.pkl, scaler.pkl, data_for_modeling.csv) are in the same directory.")
        return None, None, None, None, None


model, pca, scaler, feature_names, column_means, continuous_features = load_components()

st.markdown('<h1 class="main-header">Customer Churn Predictor</h1>', unsafe_allow_html=True)


# Stop the app if components failed to load by checking essential objects
if model is None or pca is None or scaler is None:
    st.warning("Could not load all necessary components. The app cannot continue.")
    st.stop()

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Manual Input", "Upload CSV File"])

# Manual Input Tab
with tab1:
    st.header("Predict from Manual Input")
    st.info("Input the values for a single customer. The data should be preprocessed (e.g., categorical features encoded as numbers). Default values are the mean of each feature.")

    with st.form("manual_input_form"):
        input_data = {}
        
        num_cols = 4
        
        with st.expander("Features I", expanded=True):
            cols = st.columns(num_cols)
            for i, feature in enumerate(feature_names[:50]):
                with cols[i % num_cols]:
                    input_data[feature] = st.number_input(
                        label=feature, 
                        value=column_means.get(feature, 0.0),
                        key=f"manual_{feature}"
                    )
        
        with st.expander("Features II"):
            cols = st.columns(num_cols)
            for i, feature in enumerate(feature_names[50:100]):
                with cols[i % num_cols]:
                    input_data[feature] = st.number_input(
                        label=feature,
                        value=column_means.get(feature, 0.0),
                        key=f"manual_{feature}"
                    )
                    
        with st.expander("Features III"):
            cols = st.columns(num_cols)
            for i, feature in enumerate(feature_names[100:]):
                with cols[i % num_cols]:
                    input_data[feature] = st.number_input(
                        label=feature,
                        value=column_means.get(feature, 0.0),
                        key=f"manual_{feature}"
                    )
        
        predict_button_manual = st.form_submit_button("Predict Churn")

    if predict_button_manual:
        # Create a DataFrame from the input and ensure correct column order
        input_df = pd.DataFrame([input_data])[feature_names]

        processed_df = input_df.copy()
         # Scale only the continuous features that the scaler was fitted on
        processed_df[continuous_features] = scaler.transform(input_df[continuous_features])
        
        pca_features = pca.transform(processed_df)
        
        # Predict
        prediction = model.predict(pca_features)[0]
        probability = model.predict_proba(pca_features)[0]
        churn_prob = probability[1]

        # Display result
        if prediction == 1:
            st.markdown(f'<div class="prediction-result positive-prediction">Prediction: **Churn** (Probability: {churn_prob:.2%})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-result negative-prediction">Prediction: **No Churn** (Probability of Churn: {churn_prob:.2%})</div>', unsafe_allow_html=True)

# CSV Upload Tab 
with tab2:
    st.header("Predict from CSV File")
    st.info(f"Upload a CSV file with the **{len(feature_names)} preprocessed feature columns** used for training. The app will apply scaling and PCA before predicting.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(batch_df.head())

            if st.button("Run Batch Prediction"):
                with st.spinner("Processing and predicting..."):
                    if not all(col in batch_df.columns for col in feature_names):
                        st.error("The uploaded file is missing required feature columns.")
                    else:
                        # Preprocess and predict
                        batch_features = batch_df[feature_names]
                       # Scale only the continuous features
                        batch_features[continuous_features] = scaler.transform(batch_features[continuous_features])
                        
                        # Now the entire dataframe is ready for PCA
                        pca_batch = pca.transform(batch_features)
                        
                        predictions = model.predict(pca_batch)
                        probabilities = model.predict_proba(pca_batch)[:, 1]

                        # Create results dataframe
                        results_df = batch_df.copy()
                        results_df['Predicted_Churn'] = ['Churn' if p == 1 else 'No Churn' for p in predictions]
                        results_df['Churn_Probability'] = probabilities

                        st.success("Batch prediction complete!")
                        st.markdown("### Prediction Results")
                        st.dataframe(results_df)

                        # Provide a download link
                        csv_output = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv_output,
                            file_name='churn_predictions.csv',
                            mime='text/csv',
                        )

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")