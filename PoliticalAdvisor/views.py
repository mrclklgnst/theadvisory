from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .myFAISS import respond_to_query
from PoliticalAdvisor.apps import graph, vector_store
import json
import time

def centering(request):
    return render(request, "PoliticalAdvisor/centering.html")
def index(request):
    return HttpResponse("Hello, world. You're at the Political Advisor index.")

def statement_matcher(request):
    return render(request, "PoliticalAdvisor/statementmatcher.html")

def analyze_user_input(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_input = data["message"]

            # Just to reduce the cost of OpenAI
            try:
                print('trying')
                with open("response.json", "r") as f:
                    model_output = json.load(f)
                    output = model_output
                    return JsonResponse({"message": model_output})
            except:
                print('querying openAI')
                model_output = respond_to_query(user_input, graph)
                print(model_output)
                with open("response.json", "w") as f:
                    json.dump(model_output, f, indent=2)
                answer = model_output["answer"]
                answer = json.dumps(answer, indent=2)
                return JsonResponse({"message": model_output})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

    return JsonResponse({"message": "Hello, world. Please send a POST request "})