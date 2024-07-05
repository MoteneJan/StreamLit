"""
    Simple Streamlit webserver application for serving developed classification
    models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib, os, pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Data dependencies
import pandas as pd
import numpy as np

# Load your raw data (example path)
raw_data_path = 'cleaned_test.csv'
raw_data = pd.read_csv(raw_data_path)

# Load your pickled models
models = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "SVM": "svm_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl"
}

# Load vectorizer
vectorizer_path = "vectorizers.pkl"
with open(vectorizer_path, 'rb') as file:
    vectorizers = pickle.load(file)

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# The main function where we will build the actual app
def main():
    """News Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("News Classifier")
    st.subheader("Analysing news articles")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Home", "Prediction", "Information", "Insights"]
    selection = st.sidebar.selectbox("Choose Option", options)
    
    # Building out the "Home" page
    if selection == "Home":
        st.info("Welcome to the News Classifier App")
        st.markdown("Use this application to classify news articles using different machine learning models.")
        
        # Display the dataset
        st.subheader("Dataset Overview")
        st.dataframe(raw_data.head())

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        news_text = st.text_area("Enter Text", "Type Here")

        # Adding model selection
        model_choice = st.sidebar.selectbox("Choose Machine Learning Model", list(models.keys()))

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_texts = []
            for column in vectorizers:
                vect_text = vectorizers[column].transform([news_text]).toarray()
                vect_texts.append(vect_text)
            vect_text = np.hstack(vect_texts)
 
            # Load the chosen model
            model_path = models[model_choice]
            model = load_model(model_path)

            # Make predictions
            prediction = model.predict(vect_text)

            # Mapping prediction to category name
            mapping = {0: 'business', 1: 'education', 2: 'entertainment', 3: 'sports', 4: 'technology'}
            category = mapping.get(prediction[0], "Unknown")

            # When model has successfully run, will print prediction
            st.success(f"Text Categorized as: {category}")

    # Building out the "Information" page
    elif selection == "Information":
        st.info("Information about the Models")
        st.markdown("""
        The following machine learning models are used in this application:

        - **Decision Tree:** A model that uses a tree-like graph of decisions and their possible consequences.
        - **Logistic Regression:** A statistical model that in its basic form uses a logistic function to model a binary dependent variable.
        - **Naive Bayes:** A classifier that applies Bayes' theorem with strong (naive) independence assumptions between the features.
        - **Random Forest:** An ensemble learning method that operates by constructing a multitude of decision trees.
        - **SVM (Support Vector Machine):** A supervised learning model that analyzes data for classification and regression analysis.

        You can choose any of these models from the prediction page to classify your text.
        """)
 
    # Building out the "Insights" page
    elif selection == "Insights":
        st.info("Insights about the Models")

        # Dummy performance metrics for each model (for example purposes)
        performance_metrics = {
            "Logistic Regression": 0.98,
            "SVM": 0.98,
            "Naive Bayes": 0.97,
            "Decision Tree": 0.86,
            "Random Forest": 0.96
        }

        # Display the models and the comparison button
        st.markdown("### Model Performance Comparison")
        st.markdown("""
        The performance of different machine learning models used in this application is as follows:

        - **Random Forest:** 0.96
        - **Decision Tree:** 0.86
        - **Naive Bayes:** 0.97
        - **SVM:** 0.98
        - **Logistic Regression:** 0.98            
        
        """)
        st.write("This section compares the performance of different machine learning models used in this application.")

        if st.button("Compare Models"):
            # Creating a bar chart with different colors
            fig, ax = plt.subplots()
            models_names = list(performance_metrics.keys())
            accuracies = list(performance_metrics.values())
            
            colors = sns.color_palette("hsv", len(models_names))
            ax.barh(models_names, accuracies, color=colors)
            ax.set_xlabel('Accuracy')
            ax.set_title('Model Comparison')

            # Displaying the chart
            st.pyplot(fig)

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
