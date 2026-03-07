# Fake-News-Prediction-System

## Overview

This project implements a machine learning-based Fake News Detection System that classifies news articles as Fake or True using Natural Language Processing (NLP) techniques.

The system uses TF-IDF feature extraction combined with supervised learning models such as Logistic Regression and Linear Support Vector Machine (SVM). The final model is deployed using Streamlit for real-time prediction.

The objective of this project is to design a scalable and modular solution that simulates real-world news classification using time-based data splitting.

---

## Live Demo

The project is deployed using **Streamlit** and connected to a **FastAPI prediction API**.

Try it here:  
   https://fake-news-prediction-system-6wjmq3n2utrb943frqbifd.streamlit.app/

Note: The API is deployed on Render’s free tier. If inactive, the first request may take ~50–60 seconds while the server wakes up. Subsequent requests will be faster.

Enter any news headline or article text and the system will classify it as **Fake** or **Real** using the trained ML pipeline.

---

## Problem Statement

The rapid growth of online media platforms has significantly increased the spread of misinformation. Fake news can influence public opinion, political decisions, and social stability. Manual verification is not scalable due to the large volume of digital content.

This project aims to develop an automated machine learning system that can accurately classify news articles as real or fake based on textual features.

---

## Project Structure

```
Fake-News-Detection-System/
│
├── preprocess.py      # Data cleaning and time-based splitting
├── model.py           # ML pipeline, training, and model saving
├── api.py             # Prediction interface using saved model
├── app.py             # Streamlit web application
├── model.joblib       # Saved trained model
├── requirements.txt   # Project dependencies
└── README.md
```

---

## Technologies Used

- Python
- pandas
- numpy
- scikit-learn
- TF-IDF Vectorizer
- StandardScaler
- joblib
- Streamlit

---

## System Architecture

The system follows a modular pipeline-based design:

1. Text preprocessing (cleaning, normalization)
2. Feature extraction using TF-IDF
3. Scaling of numerical features
4. Model training using Logistic Regression / Linear SVM
5. Model serialization
6. Deployment through Streamlit interface
7. The use of a Scikit-learn Pipeline ensures consistent preprocessing during both training and inference.

---

## Dataset

The dataset was sourced from:

- Kaggle – Fake and Real News Dataset
   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

After preprocessing and cleaning:

- Total records: ~39,000+
- Balanced distribution between fake and true classes
- Time-based split used to simulate real-world deployment

---

## Model Training

**Data Splitting Strategy**

Instead of random splitting, a time-based split was used:

- 80% Training
- 20% Testing

This approach prevents data leakage and better reflects real-world performance on future data.

**Algorithms Used**

- Logistic Regression
- Linear Support Vector Machine (SVM)

**Best Performing Model (SVM)**

- Accuracy: ~99.5%
- High precision and recall for both classes
- Minimal false positives and false negatives

---

## Deployment

The trained model is saved using joblib and integrated into a Streamlit web application.

**Streamlit Features**

- User input for news title and article text
- Example selection option
- Real-time prediction
- Confidence score display

To run the app locally:
```
pip install -r requirements.txt
streamlit run app.py
```

---

## Key Highlights

- Modular and maintainable project structure
- Pipeline-based ML implementation
- Time-based data splitting for realistic evaluation
- Real-time prediction interface
- Clean separation between training and inference logic

---

