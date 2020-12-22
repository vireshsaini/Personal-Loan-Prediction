#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
#import os
#import sys
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
#from PIL import Image
import streamlit as st
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
#from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
#from sklearn.svm import OneClassSVM
#from pylab import rcParams
st.set_option('deprecation.showPyplotGlobalUse', False)



# Title for the project

st.write("""
# Personal Loan Prediction!
""")

# Open and Display the image

image = Image.open('https://github.com/vireshsaini/Personal-Loan-Prediction/main/Personal-Loan-Img.png')
st.image(image, caption='ML Modeling', use_column_width=True)

# Data upload by choosing the data files form the destination
#st.subheader('Upload Dataset')
data = st.file_uploader("Upload Dataset",type=["csv","txt"])
#Buffer reset to make the file upload consistant throughout the running the program 
data.seek(0)
# Read the data file
Final_Encoded_DF = pd.read_csv(data)


# Personal Loan data summary
st.subheader('Personal Loan Data Infromation')
st.dataframe(Final_Encoded_DF)
st.subheader('Personal Loan Statical Data Analysis')
st.write(Final_Encoded_DF.describe())
chart = st.bar_chart(Final_Encoded_DF)

st.write("### Heatmap -  Co-relation between variables ")
fig, ax = plt.subplots(figsize=(10,10))
st.write(sns.heatmap(Final_Encoded_DF.corr(), annot=True,linewidths=0.5))
st.pyplot(False)

st.write("### Show Pie Chart and Value Counts of Target Columns")
st.write(Final_Encoded_DF.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
st.pyplot(False)
st.write(Final_Encoded_DF.iloc[:,-1].value_counts())

Final_Encoded_DF.hist()
#st.pyplot()

# Data spllit into Independent (X) and dependent (Y) variables
X = Final_Encoded_DF.iloc[:, 0:18].values
Y = Final_Encoded_DF.iloc[:, -1].values


# Divide data into training and testing 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)



#Get the feature input from user

def get_user_input():
    st.subheader('Selected Data  Input By User')
    Age = st.sidebar.slider('Age', 39, 65, 39)
    #ApplicantIncome = st.slider('ApplicantIncome',6840, 1199760,537708)
    LoanAmount = st.sidebar.slider('LoanAmount', 9000, 700000, 30402)
    Total_Income = st.sidebar.slider('Total_Income', 1760, 1200843, 490452)
    Loan_Amount_Term = st.sidebar.slider('Loan_Amount_Term',6, 60, 36)
    Credit_History = st.sidebar.slider('Credit_History', 0, 1, 1)
    Cibil_Score = st.sidebar.slider('Cibil_Score', 300, 850, 825)
    Adhar_Card = st.sidebar.slider('Adhar_Card',0, 1, 1)
    Gender_Female = st.sidebar.slider('Gender_Female', 0,1, 0)
    Gender_Male = st.sidebar.slider('Gender_Male', 0,1, 1)
    Married_No = st.sidebar.slider('Married_No', 0, 1, 1)
    Married_Yes = st.sidebar.slider('Married_Yes', 0, 1, 0)
    Education_Graduate = st.sidebar.slider('Education_Graduate', 0, 1, 1)
    Education_NotGraduate = st.sidebar.slider('Education_NotGraduate', 0, 1, 0)
    Self_Employed_No = st.sidebar.slider('Self_Employed_No', 0,1, 1)
    Self_Employed_Yes = st.sidebar.slider('Self_Employed_Yes',0, 1, 0)
    Application_Area_Rural = st.sidebar.slider('Application_Area_Rural', 0, 1, 0) 
    Application_Area_Semiurban = st.sidebar.slider('Application_Area_Semiurban', 0, 1, 0) 
    Application_Area_Urban = st.sidebar.slider('Application_Area_Urban', 0, 1,1)
            
# Stoer the data into variables

    user_data = {'Age' : Age,
                #'ApplicantIncome' : ApplicantIncome,
                'LoanAmount' : LoanAmount,
                'Total_Income' : Total_Income,
                'Loan_Amount_Term' : Loan_Amount_Term,
                'Credit_History' : Credit_History,
                'Cibil_Score' : Cibil_Score,
                'Adhar_Card' : Adhar_Card,
                'Gender_Female' : Gender_Female,
                'Gender_Male' : Gender_Male,
                'Married_No' : Married_No,
                'Married_Yes' : Married_Yes,
                'Education_Graduate' : Education_Graduate,
                'Education_NotGraduate' : Education_NotGraduate,
                'Self_Employed_No' : Self_Employed_No,
                'Self_Employed_Yes' : Self_Employed_Yes,
                'Application_Area_Rural' : Application_Area_Rural,
                'Application_Area_Semiurban' : Application_Area_Semiurban,
                'Application_Area_Urban' : Application_Area_Urban
                }

# Transform the data into Data frame

    features = pd.DataFrame(user_data, index = [0])
    return features
    
# Store the user input into a variable

user_input = get_user_input()

# Set a subheader and display the users input

#st.subheader('Selected User Input')
st.write(user_input)

# ML Model creation

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Show the model metrics

st.subheader('ML Model Test Accuracey Score')

st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%' )

# Store the model prediction in a variable

prediction = RandomForestClassifier.predict(user_input)

# set the subheader and display the classification

st.subheader('Classification Result ')
st.write(prediction)





