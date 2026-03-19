from django.db import models

from django.db import models


class XrayScan(models.Model):
    image = models.ImageField(upload_to="xrays_images/")
    result = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    objects = models.Manager()
