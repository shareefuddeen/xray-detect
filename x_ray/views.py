from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .service import predict_xray
from .models import XrayScan


def upload_xray(request):
    result = None
    return render(request, "index.html", {"result": result})


@csrf_exempt
def detect_xray(request):
    if request.method == "POST":
        image = request.FILES.get("xray")

        if not image:
            return JsonResponse({"error": "No image uploaded"}, status=400)

        results = predict_xray(image)

        scan = XrayScan.objects.create(image=image, result=results)

        return JsonResponse({"success": True, "predictions": results, "id": scan.id})

    return JsonResponse({"error": "Only POST allowed"}, status=405)
