'''
lab21 : Prediction de credit du logement
Realise par : KOUBACHI Yasser EMSI 5IIR G1
https://www.kaggle.com/datasets/sazid28/home-loan/data?select=test.csv
'''
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
#Step 1 : Datasets
dt = pd.read_csv('Datasets/leon.csv')
#dt = pd.read_csv('Datasets/train.csv')
#Feature engineering
print(dt.head())
print(dt.info())
print(dt.isna().sum())
print(dt.shape)
#modes : valeur plus frequent
#mediane : valeur central
def trans(data):
    for c in data.columns:
        if data[c].dtype=='int64' or data[c].dtype=='float64':
            data[c].fillna(data[c].median(),inplace=True)
        else:
              data[c].fillna(data[c].mode()[0],inplace=True)

print('/////////////////////////////')
print(dt.shape)
trans(dt)
print(trans(dt))
#trans(dt)
print(dt.isna().sum())
print('0000')
#target exploration (loan_status)
print(dt["Loan_Status"].value_counts(normalize=True)*100)
'''
figures and stuff
'''
#
# fig = px.histogram(dt,x="Loan_Status",title='Crédit accordé ou pas', color="Loan_Status",template= 'plotly_dark')
# fig.show(font = dict(size=17,family="Franklin Gothic"))
#
# #Genre
# fig = px.histogram(dt, x="Gender",title='Genre',color="Gender",template= 'plotly_dark')
# fig.show(font = dict(size=17,family="Franklin Gothic"))
# #Dependents
# fig = px.histogram(dt, x="Dependents",title='Dependents',color="Dependents",template= 'plotly_dark')
# fig.show(font = dict(size=17,family="Franklin Gothic"))
#
# fig = px.pie(dt, names="Dependents",title='Dependents',color="Dependents",template= 'plotly_dark')
# fig.show(font = dict(size=17,family="Franklin Gothic"))

# Variables Numériques
var_num=["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]
print(dt[var_num].describe())
print(dt.info())
#
# #ApplicantIncome
# fig = px.histogram(dt, x="ApplicantIncome", facet_col="Loan_Status",
#                   marginal="box",color="Loan_Status")
# fig.show()
#
# #CoapplicantIncome
# fig = px.histogram(dt, x="CoapplicantIncome", facet_col="Loan_Status",
#                   marginal="box",color="Loan_Status")
# fig.show()
#
#
# #LoanAmount
# fig = px.histogram(dt, x="LoanAmount", facet_col="Loan_Status", marginal="box",color="Loan_Status")
# fig.show()
#Analyse bivariée

dt['Credit_History'] = dt['Credit_History'].replace(1.0,'Yes')
dt['Credit_History'] = dt['Credit_History'].replace(0.0,'No')
#Les variables categoriques
var_cat=["Loan_Status","Gender","Married","Dependents","Education","Self_Employed","Property_Area","Credit_History"]

fig, axes = plt.subplots(4, 2, figsize=(12, 15))
for idx, cat_col in enumerate(var_cat):
    row, col = idx // 2, idx % 2
    sns.countplot(x=cat_col, data=dt, hue="Loan_Status", ax=axes[row, col])

plt.subplots_adjust(hspace=1)

#sns.heatmap(dt.corr(),cmap = 'Wistia', annot= True)


dt_num=dt[var_num]
dt_cat=dt[var_cat]
print(dt_cat)

dt_cat=pd.get_dummies(dt_cat,drop_first=True)
print(dt_cat)

dt_encoded=pd.concat([dt_cat,dt_num],axis=1)
print(dt_encoded)


y=dt_encoded["Loan_Status_Y"]
print(y)

x=dt_encoded.drop("Loan_Status_Y",axis=1)
print(x)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
dt_encoded.to_csv("leon.csv")
#Step 2: Model
model=LogisticRegression()
model.fit(x_train,y_train)
print("Votre Intelligence Arti est fiable à")
print(model.score(x_test,y_test)*100),print("%")
#step 3: Train

#Step 4: Test


# Streamlit for web deployment :