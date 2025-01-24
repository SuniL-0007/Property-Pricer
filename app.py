import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder

model = joblib.load("ensemble_model_best.pkl")

dataset = pd.read_csv("Final_Data.csv")

dataset = dataset.drop(columns=['price'])

def predict_house_price(input_data):
    input_df = pd.DataFrame([input_data])
    input_df['total_rooms'] = input_df['bedRoom'] + input_df['bathroom'] + input_df['balcony']
    input_df['house_age'] = input_df['agePossession']
    input_df = input_df.drop(columns=['bedRoom', 'bathroom', 'balcony', 'agePossession'])
    return input_df


st.title("House Price Prediction")

st.header("Enter House Data")
user_input = {}
for col in dataset.columns:
    user_input[col] = st.text_input(f"Select {col}")
    
if st.button("Predict Price"):

    final_input = predict_house_price(user_input)
    try:

        prediction = model.predict(final_input)
        st.write(prediction[0])
    except ValueError as e:
        st.error(f"Prediction error: {e}")
