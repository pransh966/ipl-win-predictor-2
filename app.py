import streamlit as st
import joblib
import pickle
import pandas as pd

import sklearn.compose._column_transformer
import types

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #e0f7fa, #f1f8e9);
    }
    .css-18e3th9 {
        padding-top: 1rem;
    }
    .big-font {
        font-size:22px !important;
        font-weight:600;
    }
    .center-text {
        text-align:center;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/cricket.png", width=100)
    st.title("🏏 IPL Predictor")
    st.write("Get the win probability based on match conditions.")


from pandas.io.sas.sas_constants import os_version_number_length

sklearn.compose._column_transformer._RemainderColsList = list  # Fake the missing class

with open("pipe.pkl", "rb") as f:
    pipe = pickle.load(f)

st.title('IPL Prediction App')

Teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

Cities = ['Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai',
          'Kolkata', 'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai',
          'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London',
          'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad', 'Cuttack',
          'Nagpur', 'Dharamsala', 'Kochi', 'Visakhapatnam', 'Raipur', 'Ranchi',
          'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(Teams))
with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(Teams))

selected_city = st.selectbox('Select City', sorted(Cities))
target = st.number_input('Enter Target', min_value=1)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Enter Score', min_value=0)
with col4:
    over_completed = st.number_input('Enter Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets_fallen = st.number_input('Enter Wickets Fallen', min_value=0, max_value=10)

# 👉 Make sure this is the only place where prediction logic happens
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - int(over_completed * 6)
    wickets_left = 10 - wickets_fallen
    crr = score / over_completed if over_completed > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    st.table(input_df)
    result = pipe.predict_proba(input_df)[0]
    loss=result[0]
    win=result[1]
    st.header(batting_team+"- "+str(round(win*100))+"%")
    st.header(bowling_team+"- "+str(round(loss*100))+"%")




    # Optional: Use your model to predict
    # result = pipe.predict_proba(input_df)[0]
    # st.success(f"Win Probability: {round(result[1]*100, 2)}%")
