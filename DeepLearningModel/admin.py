from django.contrib import admin
from .models import XRayImage

# Register your models here.
@admin.register(XRayImage)
class XRayImageAdmin(admin.ModelAdmin):
    pass
