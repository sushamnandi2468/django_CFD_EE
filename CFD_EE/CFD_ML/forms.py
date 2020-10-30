from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField()

class LookUpForm(forms.Form):
    first_name = forms.CharField()
    last_name = forms.CharField()
    dob = forms.DateField()

    class Meta: 
        fields = ['first_name', 'last_name', 'dob']