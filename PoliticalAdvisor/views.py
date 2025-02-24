from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .myFAISS import respond_to_query
from PoliticalAdvisor.apps import global_graph, global_vector_store, global_graph_en, global_vector_store_en
import json
import dotenv
import os
import time

dotenv.load_dotenv()
def index(request):
    return render(request, "PoliticalAdvisor/index.html")
def statement_matcher(request):
    language = request.COOKIES.get('language', 'en')
    lang_context = {
        'en': {
            'message_prompt': 'Enter a political statement that is important to you ...',
            'button_text': 'Send',
            'table_title': 'AI generated summary of party positions',
            'citations_title': 'Citations from party programs',
        },
        'de': {
            'message_prompt': 'Geben Sie eine politische Aussage ein die Ihnen wichtig ist ....',
            'button_text': 'Versenden',
            'table_title': 'AI generated summary of paty positions',
            'citations_title': 'Citations from party programs',
        }
    }
    selected_lang_context = lang_context.get(language, lang_context['en'])
    return render(request, "PoliticalAdvisor/statementmatcher.html", {"lang_context": selected_lang_context})
def analyze_user_input(request):
    if request.method == "POST":
        mockup_response = os.environ.get("MOCKUP_RESPONSE_MODE", default=False)
        data = json.loads(request.body)
        user_input = data["message"]
        language = request.COOKIES.get('language', 'de')
        if language == 'de':
            graph = global_graph
            vector_store = global_vector_store
        elif language == 'en':
            graph = global_graph_en
            vector_store = global_vector_store_en

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

