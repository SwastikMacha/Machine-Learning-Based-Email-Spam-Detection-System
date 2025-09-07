# Machine Learning-Based Email Spam Detection System

## Overview
This project is a **Machine Learning system for email spam detection**, designed to classify emails as **spam** or **ham (non-spam)**. The system leverages **text preprocessing, TF-IDF vectorization, and Naive Bayes modeling** to achieve high classification accuracy. The project is deployed as an **interactive web application using Streamlit Cloud**, allowing real-time email classification.  

## Features
- **Spam Detection:** Accurately classifies emails as spam or ham.  
- **Interactive UI:** Streamlit-based web interface for easy testing and real-time predictions.  
- **Model Persistence:** Serialized trained model and vectorizer using Pickle for reproducible deployment.  
- **Preprocessing Pipeline:** Includes tokenization, stop-word removal, and text cleaning for robust feature extraction.  

## Technologies Used
- **Programming Language:** Python  
- **Libraries:** Scikit-learn, NLTK, Pandas, NumPy, Streamlit  
- **Machine Learning Model:** Multinomial Naive Bayes  
- **Deployment:** Streamlit Cloud  
- **Version Control:** Git & GitHub  

The project uses a **[Spam Email Dataset]([https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset))** containing labeled emails classified as spam or ham (non-spam), used for training and evaluating the machine learning model.
