from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .service import predict_xray
from .models import XrayScan
import traceback
from django.contrib.auth.models import User


def upload_xray(request):
    result = None
    return render(request, "index.html", {"result": result})


@csrf_exempt
def detect_xray(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    image = request.FILES.get("xray")
    if not image:
        return JsonResponse({"error": "No image uploaded"}, status=400)

    try:
        results = predict_xray(image)

        scan = XrayScan.objects.create(image=image, result=results)

        return JsonResponse({"success": True, "predictions": results, "id": scan.id})

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": f"Error analyzing image: {str(e)}"}, status=500)


def create_admin():
    if not User.objects.filter(username="sharif").exists():
        User.objects.create_superuser(
            username="sharif", email="awalsharifpz18@gmail.com", password="admin123"
        )


def init_admin(request):
    create_admin()
    return JsonResponse({"status": "admin created"})
