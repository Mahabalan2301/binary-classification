# GenReal.ai Recruitment Task – Binary Image Classifier

## Overview

This project demonstrates an end-to-end machine learning pipeline:
- **Model Training**: Performed in Google Colab using TensorFlow/Keras, classifying cats vs dogs.
- **Cloud Deployment**: Model deployed to the cloud with Modal, enabling scalable API inference.
- **Frontend UI**: A Streamlit app (launched in Colab, using ngrok for public access) lets users upload an image and see predictions.

## Workflow

1. **Train & Export Model (`.keras` format)** in Colab:
    - Used a public cats-vs-dogs dataset.
    - Final model saved as `cat_dogs_model.keras`.

2. **Deploy Model with Modal**:
    - Placed `cat_dogs_model.keras` and `model_app.py` in the deployment folder.
    - Used Modal's Python API and built an HTTP POST endpoint for image classification.

3. **Build Functional UI with Streamlit**:
    - The Streamlit app (`app.py`) uploads images and displays results from the Modal API.
    - Publicly accessible via ngrok (Colab workaround).

## How to Run

### **Colab Training**
- Clone or copy the notebook to train and export `cat_dogs_model.keras`.

### **Modal Deployment**
- Ensure Modal Python SDK is installed (`pip install modal`).
- Authenticate with `python -m modal setup`.
- Place `cat_dogs_model.keras` and `model_app.py` in the same folder.
- Deploy with:  
