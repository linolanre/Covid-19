import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


data = requests.get('https://api.covidtracking.com/v1/states/daily.json')
data = data.json()

pd.set_option('display.max_columns', None)
v = pd.DataFrame(data)
print(v.head())

v = v[['date', 'state', 'positive', 'negative', 'totalTestResults', 'hospitalizedCurrently', 'death', 'totalTestsPeopleViral']]
v = v.rename(columns={'date': 'Date', 'state': 'State', 'positive': 'Positive Cases', 'negative': 'Negative Cases', \
                        'totalTestResults': 'Total Tests', 'hospitalizedCurrently': 'Currently Hospitalized', \
                        'death': 'Deaths', 'totalTestsPeopleViral': 'Total Tests - Viral'})

v = v.drop_duplicates()
v = v.dropna()

scaler = MinMaxScaler()

num_cols = ['Positive Cases', 'Negative Cases', 'Total Tests', 'Currently Hospitalized', 'Deaths', 'Total Tests - Viral']

v[num_cols] = scaler.fit_transform(v[num_cols])


fig = px.scatter(v, x='Date', y='Positive Cases', color='State')
fig.show()



corr = v.corr()
fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index), annotation_text=corr.round(2).values, colorscale='Viridis')
fig.show()


X = v[['Date']]
y = v['Positive Cases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

import joblib 
joblib.dump(regressor, 'reg.pkl')
