from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import json


def index(request):
    return HttpResponse("Hello, world. You're at the Political Advisor index.")

def statement_matcher(request):
    return render(request, "PoliticalAdvisor/statementmatcher.html")

def analyze_user_input(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_input = data["message"]
            return JsonResponse({"message": f"User input: {user_input}"})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

    return JsonResponse({"message": "Hello, world. Please send a POST request "})