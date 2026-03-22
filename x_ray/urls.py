from django.urls import path
from .views import upload_xray, detect_xray

urlpatterns = [path("", upload_xray), path("detect/", detect_xray)]
