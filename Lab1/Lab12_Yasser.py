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
st.header('classification des fleyrs Irise')
st.image('irisDataset.png')
#st.write(iris.data)

st.write(iris.target_names[prediction])

