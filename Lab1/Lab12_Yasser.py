# Lab12: classification des fleyrs d'irise
# Realiser par Yasser KOUBACHI EMSI 2023-2024
# import packages
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import streamlit as st

# Step1 : Dataset
iris = datasets.load_iris()
print(iris.data)
print(iris.target)
print(iris.target_names)
# Step2 : Model
model = RandomForestClassifier()
# Step3 : Train
model.fit(iris.data,iris.target)
# Step4 : Test
prediction = model.predict([[5.9,3. ,5.1,1.8]])

# Model deployment cmd to run :::  streamlit run Lab12_Yasser.py
st.header('Classification des fleyrs Irise')
def userInput():
    sepal_length = st.sidebar.slider("sepal length", 4.3, 7.9, 6.0)
    sepal_width = st.sidebar.slider("sepal width", 2.0, 4.4, 3.0)
    pepal_length = st.sidebar.slider("pepal length", 1.0, 9.2, 2.0)
    pepal_width = st.sidebar.slider("pepal width", 0.1, 2.5, 1.0)
    data = {
        'sepal_length' : sepal_length,
        'sepal_width' : sepal_width,
        'pepal_length' : pepal_length,
        'pepal_width' : pepal_width,
    }
    flower_features = pd.DataFrame(data, index=[0])
    return flower_features

df = userInput()
st.write(df)

st.subheader('iris flower prediction')
prediction = model.predict(df)
st.write(iris.target_names[prediction])

#st.image('irisDataset.png')
#st.write(iris.data)

#st.write(iris.target_names[prediction])

