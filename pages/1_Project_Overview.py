import streamlit as st

st.markdown("# Project Overview")
st.write(
    """
The purpose of this project is to accurately predict the GDP of various countries throughout the world. Understanding GDP and the factors that impact GDP is important to understanding how businesses can expand, keep investors informed, and get economic feedback from policies. Our project uses a linear regression machine learning model to accurately predict the long-term GDP of several countries, using the United States as our trial. This project aims to help fill the gap of GDP information missing in current prediction systems.

"""
)

st.markdown("# What we made")
st.write(
    """
We created GDPredict, a web application that predicts the GDP of various countries using a linear regression machine learning model. The app provides users with insights into the factors influencing GDP and offers accurate long-term predictions to aid in economic planning and decision-making.

"""
)

st.markdown("# Our Tech Stack")
st.write(
    """ We created our own datasets using data from FRED and implemented a linear regression machine learning model. The web application was built using Streamlit for an interactive user interface.
    """
)

st.markdown("# How It Works")
st.write(
    """
    1. CSV files contain real GDP data and exogenous events with their economic impact
    2. Data is cleaned and sorted and  split into training and testing sets
    3. The model is trained and recieved an R^2 and normalized RMSE score for evaluation
    4. Streamlit plots the actual, predicted, and forecasted GDP as a line graph on the webpage for the user 
    """
)

st.markdown("# Project Steps")
st.write(
    """
1. Data Collection: We gathered historical economic data from reliable sources such as FRED (Federal Reserve Economic Data) to create our datasets.
2. We trained a linear regression machine learning model first using the  historical United States data to test different models' accuracy, and we discovered that linear regression provided the best results. Our highest R^2 score was 0.99, indicating a strong correlation between the input features and GDP.
3. Web Application Development: We built the GDPredict web application using Streamlit, allowing users to interact with the model and view forecasted GDP with exogenous events applied.

"""
)

st.markdown("# Challenges Faced")
st.write(
    """
1. Data Quality: Ensuring the accuracy and completeness of the economic data collected from international databases was a challenge, as inconsistencies could affect model performance. We normalized and cleaned the data to address this issue.
2. Model Selection: Choosing the right machine learning model required extensive experimentation and evaluation to achieve the best predictive accuracy. Ultimately, linear regression was selected for its effectiveness in this context.
3. Accurate Predictions: Achieving a high R^2 score was crucial for the model's reliability. We iteratively refined our model and features to reach an R^2 score of 0.99.
4. Choosing GDP Factors: Identifying the most relevant economic indicators for each country that influence GDP was essential for model accuracy. We conducted thorough research to select these factors.
"""
)

st.markdown("# What's Next?")
st.write(
    """
1. Expand Country Coverage: We plan to extend GDPredict to include more countries, allowing users to predict GDP for a wider range of economies.
2. Enhance Model Accuracy: We aim to incorporate additional economic indicators or more complex models to improve the accuracy of our GDP predictions further.
"""
)
