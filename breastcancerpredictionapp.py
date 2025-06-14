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
    
    # Display model information for debugging
    st.info(f"Model type: {type(model).__name__}")
    if hasattr(model, 'n_features_in_'):
        st.info(f"Model expects {model.n_features_in_} features")
    
    # Create input fields for ALL 30 features of the breast cancer dataset
    st.subheader("Mean Values")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        radius_mean = st.number_input('Radius Mean', min_value=0.0, value=14.0, key='radius_mean')
        texture_mean = st.number_input('Texture Mean', min_value=0.0, value=19.0, key='texture_mean')
        perimeter_mean = st.number_input('Perimeter Mean', min_value=0.0, value=92.0, key='perimeter_mean')
        area_mean = st.number_input('Area Mean', min_value=0.0, value=655.0, key='area_mean')
        smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0, value=0.1, key='smoothness_mean')
        compactness_mean = st.number_input('Compactness Mean', min_value=0.0, value=0.1, key='compactness_mean')
        concavity_mean = st.number_input('Concavity Mean', min_value=0.0, value=0.1, key='concavity_mean')
        concave_points_mean = st.number_input('Concave Points Mean', min_value=0.0, value=0.05, key='concave_points_mean')
        symmetry_mean = st.number_input('Symmetry Mean', min_value=0.0, value=0.18, key='symmetry_mean')
        fractal_dimension_mean = st.number_input('Fractal Dimension Mean', min_value=0.0, value=0.06, key='fractal_dimension_mean')
    
    st.subheader("Standard Error Values")
    with col2:
        radius_se = st.number_input('Radius SE', min_value=0.0, value=0.4, key='radius_se')
        texture_se = st.number_input('Texture SE', min_value=0.0, value=1.2, key='texture_se')
        perimeter_se = st.number_input('Perimeter SE', min_value=0.0, value=2.9, key='perimeter_se')
        area_se = st.number_input('Area SE', min_value=0.0, value=40.0, key='area_se')
        smoothness_se = st.number_input('Smoothness SE', min_value=0.0, value=0.007, key='smoothness_se')
        compactness_se = st.number_input('Compactness SE', min_value=0.0, value=0.025, key='compactness_se')
        concavity_se = st.number_input('Concavity SE', min_value=0.0, value=0.032, key='concavity_se')
        concave_points_se = st.number_input('Concave Points SE', min_value=0.0, value=0.012, key='concave_points_se')
        symmetry_se = st.number_input('Symmetry SE', min_value=0.0, value=0.02, key='symmetry_se')
        fractal_dimension_se = st.number_input('Fractal Dimension SE', min_value=0.0, value=0.003, key='fractal_dimension_se')
    
    st.subheader("Worst Values")
    with col3:
        radius_worst = st.number_input('Radius Worst', min_value=0.0, value=16.0, key='radius_worst')
        texture_worst = st.number_input('Texture Worst', min_value=0.0, value=25.0, key='texture_worst')
        perimeter_worst = st.number_input('Perimeter Worst', min_value=0.0, value=107.0, key='perimeter_worst')
        area_worst = st.number_input('Area Worst', min_value=0.0, value=880.0, key='area_worst')
        smoothness_worst = st.number_input('Smoothness Worst', min_value=0.0, value=0.13, key='smoothness_worst')
        compactness_worst = st.number_input('Compactness Worst', min_value=0.0, value=0.25, key='compactness_worst')
        concavity_worst = st.number_input('Concavity Worst', min_value=0.0, value=0.27, key='concavity_worst')
        concave_points_worst = st.number_input('Concave Points Worst', min_value=0.0, value=0.11, key='concave_points_worst')
        symmetry_worst = st.number_input('Symmetry Worst', min_value=0.0, value=0.29, key='symmetry_worst')
        fractal_dimension_worst = st.number_input('Fractal Dimension Worst', min_value=0.0, value=0.08, key='fractal_dimension_worst')
    
    if st.button('🔍 Predict'):
        try:
            # Create input array with ALL 30 features in the correct order
            input_data = np.array([[
                radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
                radius_se, texture_se, perimeter_se, area_se, smoothness_se,
                compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
                radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
                compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
            ]])
            
            st.info(f"Input data shape: {input_data.shape}")
            
            # Debug: Check if scaler is loaded correctly
            if scaler is None:
                st.error("❌ Scaler not loaded properly!")
                st.stop()
            
            # Debug: Check scaler type
            st.info(f"Scaler type: {type(scaler).__name__}")
            
            # Scale the input data using the scaler
            try:
                scaled_data = scaler.transform(input_data)
                st.success("✅ Data scaled successfully")
                st.info(f"Scaled data shape: {scaled_data.shape}")
            except Exception as scale_error:
                st.error(f"❌ Error during scaling: {str(scale_error)}")
                st.stop()
            
            # Make prediction using the model
            try:
                prediction = model.predict(scaled_data)
                st.success("✅ Prediction made successfully")
            except Exception as pred_error:
                st.error(f"❌ Error during prediction: {str(pred_error)}")
                st.stop()
            
            # Get prediction probability if available
            probability = None
            confidence = None
            try:
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(scaled_data)
                    confidence = np.max(probability) * 100
                elif hasattr(model, 'decision_function'):
                    # For SVM, we can use decision_function
                    decision_scores = model.decision_function(scaled_data)
                    st.info(f"Decision score: {decision_scores[0]:.3f}")
            except Exception as prob_error:
                st.warning(f"Could not get probability: {str(prob_error)}")
            
            # Display results
            st.header('🎯 Prediction Results')
            
            if prediction[0] == 1:
                st.success('✅ **Benign** (Non-cancerous)')
                st.balloons()
            else:
                st.error('⚠️ **Malignant** (Cancerous)')
                st.warning('Please consult with a healthcare professional immediately.')
            
            if confidence is not None:
                st.info(f'Confidence: {confidence:.1f}%')
                
                # Show probability breakdown
                st.subheader('Probability Breakdown')
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Malignant", f"{probability[0][0]:.3f}")
                with col2:
                    st.metric("Benign", f"{probability[0][1]:.3f}")
            
        except ValueError as e:
            st.error(f"❌ ValueError: {str(e)}")
            if "Expected" in str(e) and "features" in str(e):
                st.info("🔍 **Feature count mismatch detected!**")
                st.info("The model expects a different number of features than provided.")
                st.info("Please check your model training code to see how many features were used.")
                
        except AttributeError as e:
            st.error(f"❌ AttributeError: {str(e)}")
            if "transform" in str(e):
                st.info("🔍 **Transform method issue detected!**")
                st.info("Make sure you're using scaler.transform(), not model.transform()")
                st.info("Check that your scaler.pkl file contains a valid scaler object.")
                
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            st.info("Please check that all input values are valid numbers and model files are correct.")

else:
    st.warning("⚠️ Cannot load model or scaler files.")
    st.info("Make sure 'model.pkl' and 'scaler.pkl' files are in the same directory as this script.")

# Add debugging section
st.sidebar.header("🔧 Debug Information")
if st.sidebar.button("Show Debug Info"):
    st.sidebar.subheader("Model Info")
    if model is not None:
        st.sidebar.write(f"Model type: {type(model)}")
        st.sidebar.write(f"Model attributes: {dir(model)}")
    
    st.sidebar.subheader("Scaler Info")
    if scaler is not None:
        st.sidebar.write(f"Scaler type: {type(scaler)}")
        st.sidebar.write(f"Scaler attributes: {dir(scaler)}")

# Add footer
st.markdown("---")
st.markdown("⚠️ **Disclaimer**: This tool is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.")
