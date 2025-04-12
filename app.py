import streamlit as st
import joblib
import numpy as np

#load model
model=joblib.load('model.pkl')

#ui
st.title("iris flower classifier")
st.subheader("predict the type of iris flower based on its measurements")

#Input features

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0)

sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5)

petal_length = st.slider("Petal Length (cm)", 1.0, 7.0)

petal_width = st.slider("Petal Width (cm)", 0.1, 2.5)

if st.button("Predict"):

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction =  model.predict(input_data)
    classes =  ['Setosa', 'Versicolor', 'Virginica']
    st.success(f" Prediction: {classes[prediction[0]]}")