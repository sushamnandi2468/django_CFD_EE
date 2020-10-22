from django.shortcuts import render

def home(request):
    return render(request, 'AI_Fraud_Detection/home.html')
def about(request):
    return render(request, 'AI_Fraud_Detection/about.html')
