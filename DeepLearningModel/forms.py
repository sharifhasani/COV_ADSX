from  django import forms
from .models import XRayImage

class XRayImageForm(forms.ModelForm):
    image = forms.ImageField(label="", max_length=200, required=True,
                            widget=forms.FileInput(attrs={'class': 'form-control'}))

    class Meta:
        model = XRayImage
        fields = ["image"]

