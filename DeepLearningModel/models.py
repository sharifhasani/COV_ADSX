from django.db import models

# Create your models here.
class XRayImage(models.Model):

    image = models.ImageField(verbose_name="تصویر x-ray", upload_to="uploads/x-ray", max_length=200)
    date = models.DateTimeField(verbose_name="زمان ارسال", auto_now=False, auto_now_add=True)

    class Meta:
        verbose_name = "X-Ray image"
        verbose_name_plural = "X-Ray images"

    def __str__(self):
        return self.date
