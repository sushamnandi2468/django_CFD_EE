from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import UploadFileForm
from django.core.files.storage import FileSystemStorage

from sklearn.externals import joblib
from nltk.tokenize.treebank import TreebankWordDetokenizer as wd
import matplotlib.pyplot as plt

model = joblib.load(path_to_artifacts + "./algorithms/FDTree.joblib")
wi_fn = joblib.load(path_to_artifacts + "./algorithms/wi_fn.joblib")
wi_ln = joblib.load(path_to_artifacts + "./algorithms/wi_ln.joblib")

def classify(request):
    return render(request, 'CFD_ML/model.html')

@login_required
def predict(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['document']        
        fs=FileSystemStorage()
        fs.save(uploaded_file.name, uploaded_file)
        messages.success(request, f'File uploaded successfully')
        
    context ={
        'uploaded_file' : uploaded_file
    }
    return render(request, 'CFD_ML/model.html', context)