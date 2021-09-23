from django.shortcuts import render
from django.views import View

class indexView(View):
    def get(self, request, *args, **kwargs):
        return render(request, "index.html")

    def post(self, request, *args, **kwargs):
        return HttpResponse('POST request!')
# Create your views here.
