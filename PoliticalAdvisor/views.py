from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .myFAISS import respond_to_query
from django.apps import apps
import json
import dotenv
import os
import logging

# Get the logger
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# Get the PoliticalAdvisor app config dynamically
def get_faiss_objects():
    app_config = apps.get_app_config("PoliticalAdvisor")  # Fetch app instance
    return app_config.global_graph, app_config.global_vector_store, app_config.global_graph_en, app_config.global_vector_store_en

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
            'table_title': 'KI generierte Zusammenfassung der Parteistandpunkte',
            'citations_title': 'Zitate aus Parteiprogrammen',
        }
    }
    selected_lang_context = lang_context.get(language, lang_context['en'])
    return render(request, "PoliticalAdvisor/statementmatcher.html", {"lang_context": selected_lang_context})
def analyze_user_input(request):
    graph_de, vector_store, graph_en, vector_store_en = get_faiss_objects()
    if request.method == "POST":
        mockup_response = os.environ.get("MOCKUP_RESPONSE_MODE", default=False)
        data = json.loads(request.body)
        user_input = data["message"]
        language = request.COOKIES.get('language', 'de')
        if language == 'de':
            graph = graph_de

        elif language == 'en':
            graph = graph_en


        if mockup_response == "True":

            try:
                # try to load local JSON
                with open("response.json", "r") as f:
                    model_output = json.load(f)
                    logger.info("Local response found and loaded")
                    return JsonResponse({"message": model_output})

            except:
                logger.info("No local response found, querying OpenAI")
                model_output = respond_to_query(user_input, graph)

                # check if answer is a dictionary
                if isinstance(model_output["answer"], dict):
                    return JsonResponse({"message": model_output})

                else:
                    # else try to reformat the answer
                    try:
                        dict_cleaned = dict(
                            model_output["answer"].replace("```json\n", "").replace("```", "").strip())
                        model_output["answer"] = dict_cleaned
                        with open("response.json", "w") as f:
                            json.dump(model_output, f, indent=2)
                        return JsonResponse({"message": model_output})
                    # if not possible return the original answer
                    except:
                        logger.info('Response from OpenAI not in expected format')
                        return JsonResponse({"message": model_output})
        else:
            try:
                # query openAI, returns dict with keys 'answer' and 'citations'
                model_output = respond_to_query(user_input, graph)
                # check if answer in needed format
                if isinstance(model_output["answer"], dict):
                    return JsonResponse({"message": model_output})
                # else try to reformat the answer
                else:
                    try:
                        dict_cleaned = dict(model_output["answer"].replace("```json\n", "").replace("```", "").strip())
                        model_output["answer"] = dict_cleaned
                        return JsonResponse({"message": model_output})
                    except:
                        logger.info('Response from OpenAI not in expected format')
                        return JsonResponse({"message": model_output})
            except:
                logger.info("No response from OpenAI")
                return JsonResponse({"message": 'No response from OpenAI'}, status=400)

    else:
        # if not a POST request return error
        logger.info("No POST request")
        return JsonResponse({"message": "No POST request"}, status=400)