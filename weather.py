import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""    
# Weather Predictor App

This app predicts the **Weather** using a set of parameters
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    precipitation = st.sidebar.slider('Precipitation', 0.0, 55.9, 0.0)
    temp_max = st.sidebar.slider('Max Temperature', -1.6, 35.6, 0.0)
    temp_min = st.sidebar.slider('Min Temperature', -7.0, 18.3, 0.0)
    wind = st.sidebar.slider('Wind', 0.4, 9.5, 0.0)
    data = {'precipitation': precipitation,
            'temp_max': temp_max,
            'temp_min': temp_min,
            'wind': wind}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

rain = pd.read_csv('https://raw.githubusercontent.com/Amirhakimhssn/Weather/main/seattle-weather(nodate).csv')
X = rain.drop('weather', axis=1)
Y = rain['weather']

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(Y.unique())

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(['drizzle', 'rain', 'sun', 'snow', 'fog'],prediction_proba)
