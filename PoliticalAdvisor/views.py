from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .myFAISS import respond_to_query
from PoliticalAdvisor.apps import graph, vector_store
import json
import time


def index(request):
    return HttpResponse("Hello, world. You're at the Political Advisor index.")

def statement_matcher(request):
    return render(request, "PoliticalAdvisor/statementmatcher.html")

def analyze_user_input(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_input = data["message"]
            model_output = respond_to_query(user_input, graph)
            answer = model_output["citations"]
            answer = json.dumps(answer, indent=2)
            print(answer)
            return JsonResponse({"message": f"{answer}"})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

    return JsonResponse({"message": "Hello, world. Please send a POST request "})