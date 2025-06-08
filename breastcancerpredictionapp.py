import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Your current Streamlit app with improved error handling
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'model.pkl' not found. Please upload the trained model file.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_scaler():
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except FileNotFoundError:
        st.error("❌ Scaler file 'scaler.pkl' not found. Please upload the scaler file.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading scaler: {str(e)}")
        return None

# Alternative: Load combined file
@st.cache_resource
def load_combined_model():
    try:
        with open('breast_cancer_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ Combined model file not found.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading combined model: {str(e)}")
        return None

# Load model and scaler
model = load_model()
scaler = load_scaler()

# Alternative approach:
# combined_model = load_combined_model()
# if combined_model:
#     model = combined_model['model']
#     scaler = combined_model['scaler']
#     feature_names = combined_model['feature_names']

# App title and description
st.title('🔬 Breast Cancer Prediction')
st.write('Enter the required measurements to predict breast cancer diagnosis')

# Only show the input form if both model and scaler are loaded
if model is not None and scaler is not None:
    # Create input fields for features
    st.header('Patient Data Input')
    
    # Example feature inputs (adjust based on your actual features)
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
    
    # Add more features as needed for your specific model...
    # This is just an example with 10 features
    
    if st.button('🔍 Predict'):
        try:
            # Prepare input data
            input_data = np.array([[
                radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean
                # Add more features to match your model's expected input
            ]])
            
            # Scale the input data
            scaled_data = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(scaled_data)
            
            # Get prediction probability if available
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
                st.warning('Please consult with a healthcare professional immediately.')
            
            if confidence:
                st.info(f'Confidence: {confidence:.1f}%')
                
                # Show probability breakdown
                if hasattr(model, 'predict_proba'):
                    st.subheader('Probability Breakdown')
                    prob_df = pd.DataFrame({
                        'Diagnosis': ['Malignant', 'Benign'],
                        'Probability': [f"{probability[0][0]:.3f}", f"{probability[0][1]:.3f}"]
                    })
                    st.dataframe(prob_df)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please check that all input values are valid numbers.")

else:
    st.warning("⚠️ Cannot load model or scaler files. Please ensure both 'model.pkl' and 'scaler.pkl' are uploaded to your repository.")
    
    with st.expander("📋 Instructions to Fix This Issue"):
        st.markdown("""
        1. **Train your model** and save both the model and scaler:
           ```python
           # Save model
           with open('model.pkl', 'wb') as f:
               pickle.dump(trained_model, f)
           
           # Save scaler
           with open('scaler.pkl', 'wb') as f:
               pickle.dump(fitted_scaler, f)
           ```
        
        2. **Upload files** to your GitHub repository:
           - `model.pkl` - Your trained machine learning model
           - `scaler.pkl` - The fitted scaler used during training
        
        3. **Push changes** to GitHub and redeploy your Streamlit app
        """)

# Add footer
st.markdown("---")
st.markdown("⚠️ **Disclaimer**: This tool is for educational purposes only and should not be used as a substitute for professional medical advice.")