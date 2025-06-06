![banner](images/Default_Prediction.png)
Banner [source](https://banner.godori.dev/)

![Python version](https://img.shields.io/badge/Python%20version-3.10%2B-lightgrey)
![GitHub last commit](https://img.shields.io/github/last-commit/lazziemapfurira/Loan-default-prediction-app)
![GitHub repo size](https://img.shields.io/github/repo-size/lazziemapfurira/Loan-default-prediction-app)
![Type of ML](https://img.shields.io/badge/Type%20of%20ML-Binary%20Classification-red)
![License](https://img.shields.io/badge/License-MIT-green)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/lazziemapfurira/Loan-default-prediction-app/main/streamli_app.py)
[![Open Source Love](https://badges.frapsoft.com/os/v3/open-source-200x33.png?v=103)](https://github.com/ellerbrock/open-source-badges/)
Badge [source](https://shields.io/)

# 🏦 Default Risk Prediction App

A Streamlit web application that predicts loan default risk using machine learning models (XGBoost and LightGBM) with explainable AI (SHAP) capabilities.

## App Preview

The application features a clean, user-friendly interface with multiple input sections for comprehensive risk assessment:

#![Default Risk Prediction App Screenshot](images/default_image.png)
- Interactive web interface showing the prediction form with personal information, financial details, and SHAP explanation visualization*

## Key Features

- **Dual Model Architecture**: Switch between XGBoost and LightGBM models
- **50+ Predictive Features**: Comprehensive financial and behavioral inputs
- **Explainable AI**: Interactive SHAP waterfall plots
- **Real-Time Predictions**: Instant risk assessment with probability scores
- **Responsive Design**: Works on desktop and mobile devices

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://loandefaultalternative.streamlit.app/)

## Installation

```bash
# Clone repository
git clone https://github.com/lazziemapfurira/Loan-default-prediction-app.git
cd Loan-default-prediction-app

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py


