from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .myFAISS import respond_to_query
from PoliticalAdvisor.apps import graph, vector_store
import json
import dotenv
import os
import time

dotenv.load_dotenv()

def centering(request):
    return render(request, "PoliticalAdvisor/centering.html")
def index(request):
    return HttpResponse("Hello, world. You're at the Political Advisor index.")

def statement_matcher(request):
    return render(request, "PoliticalAdvisor/statementmatcher.html")

def analyze_user_input(request):
    if request.method == "POST":
        mockup_response = os.environ.get("MOCKUP_RESPONSE_MODE", default=False)
        print(mockup_response)
        data = json.loads(request.body)
        user_input = data["message"]
        if mockup_response == True:
            try:
                # try to load the local JSON response, otherwise query OpenAI
                try:
                    with open("response.json", "r") as f:
                        model_output = json.load(f)
                        return JsonResponse({"message": model_output})
                except:
                    model_output = respond_to_query(user_input, graph)
                    with open("response.json", "w") as f:
                        json.dump(model_output, f, indent=2)
                    return JsonResponse({"message": model_output})
            except json.JSONDecodeError:
                return JsonResponse({"error": "Invalid JSON"}, status=400)
        else:
            model_output = respond_to_query(user_input, graph)
            return JsonResponse({"message": model_output})

