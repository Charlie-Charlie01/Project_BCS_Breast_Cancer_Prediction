import streamlit as st
import pickle
import numpy as np
import pandas as pd

@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'model.pkl' not found.")
        return None

@st.cache_resource
def load_scaler():
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except FileNotFoundError:
        st.error("❌ Scaler file 'scaler.pkl' not found.")
        return None

# Load model and scaler
model = load_model()
scaler = load_scaler()

st.title('🔬 Breast Cancer Prediction')

if model is not None and scaler is not None:
    st.header('Patient Data Input')
    
    # Create input fields (adjust these to match your actual features)
    col1, col2 = st.columns(2)
    
    with col1:
        radius_mean = st.number_input('Radius Mean', min_value=0.0, value=14.0)
        texture_mean = st.number_input('Texture Mean', min_value=0.0, value=19.0)
        perimeter_mean = st.number_input('Perimeter Mean', min_value=0.0, value=92.0)
        area_mean = st.number_input('Area Mean', min_value=0.0, value=655.0)
        smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0, value=0.1)
        
    with col2:
        compactness_mean = st.number_input('Compactness Mean', min_value=0.0, value=0.1)
        concavity_mean = st.number_input('Concavity Mean', min_value=0.0, value=0.1)
        concave_points_mean = st.number_input('Concave Points Mean', min_value=0.0, value=0.05)
        symmetry_mean = st.number_input('Symmetry Mean', min_value=0.0, value=0.18)
        fractal_dimension_mean = st.number_input('Fractal Dimension Mean', min_value=0.0, value=0.06)
    
    # Add more input fields to match your model's expected features
    # If your model expects 30 features (like the breast cancer dataset), add them all
    
    if st.button('🔍 Predict'):
        try:
            # Create input array - MAKE SURE this matches your training features exactly
            input_data = np.array([[
                radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean
                # Add ALL the features your model was trained on
                # If you trained on 30 features, you need all 30 here
            ]])
            
            # CORRECT: Use scaler.transform() - NOT model.transform()
            scaled_data = scaler.transform(input_data)
            
            # CORRECT: Use model.predict() on the scaled data
            prediction = model.predict(scaled_data)
            
            # Get prediction probability if the model supports it
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(scaled_data)
                confidence = np.max(probability) * 100
            else:
                confidence = None
            
            # Display results
            st.header('🎯 Prediction Results')
            
            if prediction[0] == 1:
                st.success('✅ **Benign** (Non-cancerous)')
                st.balloons()
            else:
                st.error('⚠️ **Malignant** (Cancerous)')
                st.warning('Please consult with a healthcare professional.')
            
            if confidence:
                st.info(f'Confidence: {confidence:.1f}%')
                
                # Show probability breakdown
                st.subheader('Probability Breakdown')
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Malignant", f"{probability[0][0]:.3f}")
                with col2:
                    st.metric("Benign", f"{probability[0][1]:.3f}")
            
        except ValueError as e:
            if "Expected" in str(e) and "features" in str(e):
                st.error("❌ Feature count mismatch!")
                st.info("The number of input features doesn't match what the model expects.")
                st.info("Check that you're providing all the features the model was trained on.")
            else:
                st.error(f"❌ ValueError: {str(e)}")
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please check that all input values are valid numbers.")

else:
    st.warning("⚠️ Cannot load model or scaler files.")

# Debug information (optional - remove in production)
if st.checkbox("Show Debug Info"):
    if model is not None:
        st.write(f"**Model type:** {type(model).__name__}")
    if scaler is not None:
        st.write(f"**Scaler type:** {type(scaler).__name__}")
        st.write(f"**Expected features:** {len(scaler.mean_)} features")
        st.write(f"**Scaler mean (first 5):** {scaler.mean_[:5]}")
