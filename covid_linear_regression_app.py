import streamlit as st
import pandas as pd
import numpy as np
from covid_linear_regression import regressor

st.title('Predicting Future COVID-19 Cases')
st.image('covidre.png')
st.write('Enter a future date to predict the number of positive COVID-19 cases:')
date = st.date_input('Date')

prediction = regressor.predict(np.array(25/5/2025).reshape(-1,1))[0]
st.write('Predicted number of positive cases:', prediction)