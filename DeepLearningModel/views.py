from django.shortcuts import render
from django.views import View
from .forms import XRayImageForm
from .ml import predict

class indexView(View):
    def get(self, request, *args, **kwargs):
        context = {
            'form' : XRayImageForm()
        }
        return render(request, "index.html", context)

    def post(self, request, *args, **kwargs):
        form = XRayImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            res = predict(form.cleaned_data["image"].name)
            print(form.cleaned_data["image"].name, res)
            context = {
                "result": res, 
                "image_address": form.cleaned_data["image"].name,
                "form": form
            }
            return render(request, "index.html", context)
# Create your views here.
