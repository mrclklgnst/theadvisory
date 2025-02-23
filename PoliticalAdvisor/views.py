from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .myFAISS import respond_to_query
from PoliticalAdvisor.apps import graph, vector_store
import json
import dotenv
import os
import time

dotenv.load_dotenv()
def index(request):
    return render(request, "PoliticalAdvisor/index.html")
def statement_matcher(request):
    language = request.COOKIES.get('language', 'en')
    print(language)
    return render(request, "PoliticalAdvisor/statementmatcher.html")
def analyze_user_input(request):
    if request.method == "POST":
        mockup_response = os.environ.get("MOCKUP_RESPONSE_MODE", default=False)
        data = json.loads(request.body)
        user_input = data["message"]

        if mockup_response == "True":
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
            try:
                # query openAI, returns dict with keys 'answer' and 'citations'
                model_output = respond_to_query(user_input, graph)
                if isinstance(model_output["answer"], dict):
                    return JsonResponse({"message": model_output})
                else:
                    try:
                        dict_cleaned = dict(model_output["answer"].replace("```json\n", "").replace("```", "").strip())
                        model_output["answer"] = dict_cleaned
                        return JsonResponse({"message": model_output})
                    except:
                        return JsonResponse({"message": model_output})

            except:
                return JsonResponse({"message": 'No response from OpenAI'}, status=400)

