# Breast Cancer Prediction with SVM - Streamlit App
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load saved model and scaler (replace paths if needed)
@st.cache_resource
def load_model():
    with open('breast_cancer_svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_scaler():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

model = load_model()
scaler = load_scaler()

# App title and description
st.title('Breast Cancer Prediction')
st.markdown("""
**Predict malignant/benign tumors** using SVM classifier trained on breast cancer features
""")

# Feature descriptions for user reference
feature_descriptions = {
    'mean_radius': "Mean distance from center to perimeter points",
    'mean_texture': "Standard deviation of gray-scale values",
    'mean_perimeter': "Perimeter measurement",
    'mean_area': "Area measurement",
    'mean_smoothness': "Local variation in radius lengths",
    # Add other features here...
}

# Tab interface
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

with tab1:
    st.subheader("Patient Features")
    col1, col2, col3 = st.columns(3)
    
    # Create input fields organized in columns
    with col1:
        mean_radius = st.slider('Mean Radius', 5.0, 30.0, 15.0)
        mean_texture = st.slider('Mean Texture', 5.0, 40.0, 20.0)
        mean_perimeter = st.slider('Mean Perimeter', 40.0, 200.0, 90.0)
        mean_area = st.slider('Mean Area', 150.0, 2500.0, 700.0)
        mean_smoothness = st.slider('Mean Smoothness', 0.01, 0.20, 0.08)
    
    with col2:
        mean_compactness = st.slider('Mean Compactness', 0.01, 0.5, 0.1)
        mean_concavity = st.slider('Mean Concavity', 0.0, 0.5, 0.05)
        mean_concave_points = st.slider('Mean Concave Points', 0.0, 0.2, 0.02)
        mean_symmetry = st.slider('Mean Symmetry', 0.05, 0.3, 0.15)
        mean_fractal_dim = st.slider('Mean Fractal Dim', 0.01, 0.1, 0.05)
    
    with col3:
        # Add worst features here following same pattern
        # worst_radius = st.slider(...)
        # ... (include all 30 features from your model)
        st.info("Add remaining features following column pattern")

    # Prediction button
    if st.button('Predict Diagnosis'):
        # Create input array (ensure correct feature order!)
        input_data = np.array([[mean_radius, mean_texture, mean_perimeter,
                              mean_area, mean_smoothness, mean_compactness,
                              mean_concavity, mean_concave_points, mean_symmetry,
                              mean_fractal_dim]])  # Add all 30 features
        
        # Preprocess and predict
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        proba = model.predict_proba(scaled_data)[0]
        
        # Display results
        diagnosis = "Malignant" if prediction[0] == 0 else "Benign"
        st.subheader("Prediction Result")
        st.metric(label="Diagnosis", value=diagnosis)
        st.progress(proba[0] if prediction[0] == 0 else proba[1])
        st.caption(f"Confidence: {max(proba)*100:.1f}%")

        # Interpretation
        if prediction[0] == 0:
            st.warning("Clinical follow-up recommended")
        else:
            st.success("Low risk detected")

with tab2:
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File successfully loaded")
            st.dataframe(df.head(3))
            
            if st.button('Predict Batch'):
                # Validate features
                required_features = [...]  # Add your 30 feature names
                
                if set(required_features).issubset(df.columns):
                    # Preprocess and predict
                    X = df[required_features]
                    X_scaled = scaler.transform(X)
                    predictions = model.predict(X_scaled)
                    probas = model.predict_proba(X_scaled)
                    
                    # Add results to dataframe
                    df['Prediction'] = ['Malignant' if p == 0 else 'Benign' for p in predictions]
                    df['Confidence'] = [max(proba)*100 for proba in probas]
                    
                    # Display results
                    st.subheader("Predictions")
                    st.dataframe(df)
                    
                    # Download results
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name='cancer_predictions.csv',
                        mime='text/csv'
                    )
                else:
                    st.error(f"Missing required features. Needed: {required_features}")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Add feature descriptions expander
with st.expander("Feature Descriptions"):
    for feature, description in feature_descriptions.items():
        st.markdown(f"**{feature.replace('_', ' ').title()}**: {description}")

# Footer
st.divider()
st.caption("Note: This tool provides predictive insights but should not replace clinical judgment")