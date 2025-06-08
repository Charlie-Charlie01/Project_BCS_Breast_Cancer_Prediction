# Breast Cancer Prediction Using Support Vector Machine (SVM)

## Project Overview
This project implements a machine learning solution to predict whether a breast tumor is malignant (cancerous) or benign (non-cancerous) using Support Vector Machine (SVM) algorithms. The model assists in medical diagnosis by analyzing diagnostic features extracted from breast mass imaging data.

## Problem Statement
Develop an accurate machine learning model to classify breast tumors as malignant or benign based on diagnostic features, with the primary goal of minimizing false negatives (Type 2 errors) to avoid missing cancer diagnoses while maintaining high overall accuracy.

## Dataset
- **Source**: Wisconsin Breast Cancer Dataset (sklearn.datasets)
- **Features**: 30 numerical features extracted from digitized images of breast mass
- **Target Classes**: 
  - `0`: Malignant (Cancerous)
  - `1`: Benign (Non-cancerous)
- **Total Samples**: 569 instances

### Key Features Include:
- Radius, texture, perimeter, area, smoothness
- Compactness, concavity, concave points
- Symmetry, fractal dimension
- Mean, standard error, and worst values for each measurement

## Clinical Context
### Breast Cancer Diagnosis Procedure:
1. **Initial Detection**: Mammography screening, self-examination
2. **Diagnostic Imaging**: Mammogram, ultrasound, MRI
3. **Tissue Sampling**: Core needle biopsy for definitive diagnosis
4. **Pathological Analysis**: Microscopic examination of tissue samples

## Technical Implementation

### Data Preprocessing
- **Train-Test Split**: 80-20 split with stratification

### Model Selection
- **Algorithm**: Support Vector Machine (SVM)

## Error Analysis
### Type 1 Error (False Positive):
- **Definition**: Predicting malignant when actually benign
- **Impact**: Unnecessary anxiety, additional tests, increased costs
- **Acceptable**: Less critical than missing cancer

### Type 2 Error (False Negative):
- **Definition**: Predicting benign when actually malignant
- **Impact**: **CRITICAL** - Missing cancer diagnosis, delayed treatment
- **Priority**: Minimize at all costs

## Performance Metrics
### Primary Metrics:
- **Sensitivity/Recall**: Ability to correctly identify cancer cases (minimize Type 2 errors)
- **Specificity**: Ability to correctly identify benign cases (minimize Type 1 errors)
- **Precision**: When predicting cancer, how often correct
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Overall model performance across thresholds

### Target Performance:
- **Sensitivity**: >95% (critical for cancer detection)
- **Overall Accuracy**: >90%
- **Specificity**: >85% (minimize false alarms)

## Future Improvements
- **Ensemble Methods**: Combine SVM with other algorithms
- **Feature Selection**: Identify most discriminative features
- **Deep Learning**: Explore neural network architectures
- **Clinical Integration**: Develop user-friendly interface for healthcare providers
- **Real-time Processing**: Optimize for immediate diagnostic support

## Medical Disclaimer
This model is designed for research and educational purposes. It should not be used as a sole diagnostic tool. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Ojo Gbenga Charles
- Contact: gbe01nga@gmail.com
  
## Acknowledgments
- Wisconsin Breast Cancer Dataset creators
- Scikit-learn development team
- Medical professionals who provided domain expertise
- Open-source community for tool development

---

**Note**: This project demonstrates the application of machine learning in healthcare. The model's predictions should always be validated by medical professionals before making clinical decisions.
