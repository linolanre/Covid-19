import streamlit as st
import pandas as pd
import numpy as np


st.title('Predicting Future COVID-19 Cases')
st.image('covidre.png')
st.write('Enter a future date to predict the number of positive COVID-19 cases:')
date = st.date_input('Date')
st.button('Submit')
import joblib
regressor = joblib.load(open('reg.pkl', 'rb'))

prediction = regressor.predict(np.array(24/5/2024).reshape(-1,1))[0]
st.write('Predicted number of positive cases:', prediction)

