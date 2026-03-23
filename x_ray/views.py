from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .service import predict_xray
from .models import XrayScan
import traceback


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
        # Run prediction
        results = predict_xray(image)

        # Save to database
        scan = XrayScan.objects.create(image=image, result=results)

        return JsonResponse({"success": True, "predictions": results, "id": scan.id})

    except Exception as e:
        # Log full traceback to console / Render logs
        traceback.print_exc()
        # Return error message to frontend
        return JsonResponse({"error": f"Error analyzing image: {str(e)}"}, status=500)
