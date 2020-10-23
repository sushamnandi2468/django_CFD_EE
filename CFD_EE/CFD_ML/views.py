from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import UploadFileForm
from django.core.files.storage import FileSystemStorage

from sklearn.externals import joblib
from nltk.tokenize.treebank import TreebankWordDetokenizer as wd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

model = joblib.load("CFD_ML/algorithms/FDTree.joblib")
wi_fn = joblib.load("CFD_ML/algorithms/wi_fn.joblib")
wi_ln = joblib.load("CFD_ML/algorithms/wi_ln.joblib")

def classify(request):
    return render(request, 'CFD_ML/model.html')

@login_required
def prediction(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['document']        
        fs=FileSystemStorage()
        fs.save(uploaded_file.name, uploaded_file)
        messages.success(request, f'File uploaded successfully')
        user_input = pd.read_csv('media/' + uploaded_file.name)
        prediction=compute_prediction(user_input)
        json_records= prediction.reset_index().to_json(orient = 'records')
        data = []
        data =json.loads(json_records)
        
    context ={
        'uploaded_file' : uploaded_file,
        'prediction' : data
    }
    return render(request, 'CFD_ML/model.html', context)

#Complete processing of Machine Learning model
#Identify New Customers
def new_customer_identification(input_data):
    fname_model_list = list(wi_fn.keys())
    lname_model_list = list(wi_ln.keys())
    input_data["First_Name"] = input_data["First_Name"].str.lower()
    input_data["Last_Name"] = input_data["Last_Name"].str.lower()
    input_data['Dedup'] = input_data.First_Name.isin(fname_model_list).astype(int)
    input_data['Dedup'] = input_data.Last_Name.isin(lname_model_list).astype(int)

    return input_data   

#Data Pre processing
def preprocessing(input_data):
    #input_data["First_Name"] = input_data["First_Name"].str.lower()
    #input_data["Last_Name"] = input_data["Last_Name"].str.lower()
    #DOB to be split into DD MM and YYYY for ML algo
    input_data[['DD','MM','YYYY']]=input_data.DOB.str.split("-", expand=True,)
    input_data['DD']=input_data['DD'].astype(int)
    input_data['MM']=input_data['MM'].astype(int)
    input_data['YYYY']=input_data['YYYY'].astype(int)
    #Now DOB column can be dropped from the dataframe
    input_data=input_data.drop(columns=['DOB','Customer_Type', 'PAN', 'Deceased_Flag', 'Gender', 'Martial_Status', 'PEP_Flag', 'CTF_Flag', 'Country_of_residence', 'Country_of_Origin'])

    input_data = new_customer_identification(input_data)

    try:       
        input_data = input_data.replace({"First_Name" : wi_fn})
        input_data = input_data.replace({"Last_Name" : wi_ln})
        #st.write("Names replaced")
    except Exception:
        return {"status": "Error", "message": "Error in conversion"}
    
    #cols = list(input_data.columns)
    #cols = [cols[-1]] + cols[:-1]
    #input_data=input_data[cols]
    #st.write(input_data)

    return input_data

#user_input_pr=preprocessing(user_input)
def predict(input_data):
    return model.predict_proba(input_data)
        
def postprocessing(input_data):
    if input_data[1] == 0:
        label = 'False Positive'
    else :
        label = 'Fraud'
    return {"probability": input_data[1], "label": label, "status": "OK"}
        
def compute_prediction(input_data):
    actual_data = input_data
    input_data = preprocessing(input_data)
    pred_full={}
    for i in input_data.index:
        #st.write(predict(input_data[i:i+1]))
        if input_data.at[i,'Dedup'] == 1:
            prediction = predict(input_data.iloc[i:i+1, :-1])[0]  # for the complete file
            #st.write("Prediction is", prediction)
            prediction = postprocessing(prediction)
            #st.write("Prediction post processing is", prediction)
            pred_full.update({ i : prediction['label']})
        else :
            label = 'New Customer'
            prediction= {"probability": 2, "label": label, "status": "OK"}
            pred_full.update({ i : prediction['label']})

    #st.write("Final Dict", pred_full)    
    df_pred=pd.DataFrame(list(pred_full.items()), columns=['id','label'])            
    df_pred.drop(columns='id')
    #st.write("Latest Prediction post processing is", df_pred)        
    output_data = pd.concat([input_data, df_pred.reindex(input_data.index)], axis=1)
    output_data['First_Name']=actual_data['First_Name']
    output_data['Last_Name']=actual_data['Last_Name']
    #output_data = output_data.drop(columns='Dedup')

    return output_data