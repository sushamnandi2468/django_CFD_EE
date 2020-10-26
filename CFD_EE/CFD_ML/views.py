from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import UploadFileForm
from django.core.files.storage import FileSystemStorage
import pandas as pd
import numpy as np
import json
from .algo import CustomerFraudDetection
from .forms import LookUpForm
import matplotlib.pyplot as plt
from io import StringIO
import urllib, base64

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
        p=CustomerFraudDetection(user_input)
        #prediction=cfd.compute_prediction(user_input)
        prediction=p.compute_prediction()
        #Save to CSV
        prediction.to_csv('media/output.csv')
        json_records= prediction.reset_index().to_json(orient = 'records')
        data = []
        data =json.loads(json_records)
        vals=prediction.groupby('label')['First_Name'].nunique()
        vals_series=vals.rename('Classification')
        tot_count = vals_series.sum()
        fp_count = vals_series.iloc[0:1].sum()
        f_count = vals_series.iloc[1:2].sum()
        n_count = vals_series.iloc[2:3].sum()
        fp_percent= (fp_count/tot_count)*100
        vals=vals.to_dict()
        #x= vals.keys()
        #y= vals.values()
        plt.title('Number of False Positives Captured')
        plt.xlabel('Classification')
        plt.ylabel('Count')
        #plt.plot(x,y)
        #plt.savefig('/media/graph.png')
        plt.bar(range(len(vals)), list(vals.values()), align="center")
        plt.xticks(range(len(vals)), list(vals.keys()))
        plt.savefig('media/graph.png')

    context ={
        'uploaded_file' : uploaded_file,
        'prediction' : data,
        'fp_percent' : fp_percent,
        'fp_count' : fp_count,
        'f_count' : f_count,
        'n_count' : n_count        
    }
    return render(request, 'CFD_ML/model.html', context)

def lookup(request):    
    return render(request, 'CFD_ML/lookup.html')

def singlelookup(request):    
    #if l_form.is_valid():
    #    l_form=LookUpForm()
    #    p=CustomerFraudDetection(l_form)
    #    prediction=p.lookup_compute_prediction()
    
    #context ={
    #    'prediction' : prediction
    #}
    if request.method == 'POST':
        fname = request.POST["fname"]
        lname = request.POST["lname"]
        dob = request.POST["dob"]

        input_data = {
            'First_Name' : fname,
            'Last_Name' : lname,
            'DOB' : dob
        }    
        p=CustomerFraudDetection(input_data)
        prediction=p.lookup_compute_prediction()

    context ={
        'input_data' : input_data,
        'prediction' : prediction
    }
    return render(request, 'CFD_ML/lookup.html', context)
