from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from asgiref.sync import sync_to_async
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
def sidebar(request):
    return render(request, "PoliticalAdvisor/sidebar.html")
def get_faiss_objects():
    app_config = apps.get_app_config("PoliticalAdvisor")  # Fetch app instance
    return app_config.global_graph, app_config.global_vector_store, app_config.global_graph_en, app_config.global_vector_store_en

def index(request):
    return render(request, "PoliticalAdvisor/index.html")
def electionadvisor(request):
    language = request.COOKIES.get('language', 'en')
    lang_context = {
        'en': {
            'message_prompt': 'Enter a political statement that is important to you ...',
            'button_text': 'Send',
            'table_title': 'Find below an AI generated summary of party positions and relevant citations from party programs',
            'citations_title': 'Citations from party programs',
        },
        'de': {
            'message_prompt': 'Geben Sie eine politische Aussage ein die Ihnen wichtig ist ....',
            'button_text': 'Versenden',
            'table_title': 'Finden Sie unten eine KI generierte Zusammenfassung der Parteistandpunkte und relevante Zitate aus den Partei Programmen',
            'citations_title': 'Zitate aus Parteiprogrammen',
        }
    }
    selected_lang_context = lang_context.get(language, lang_context['en'])
    return render(request, "PoliticalAdvisor/electionadvisor.html", {"lang_context": selected_lang_context})
async def analyze_user_input(request):
    graph_de, vector_store, graph_en, vector_store_en = get_faiss_objects()

    if request.method == "POST":
        mockup_response = os.environ.get("MOCKUP_RESPONSE_MODE", default=False)
        data = json.loads(request.body)
        user_input = data["message"]
        language = request.COOKIES.get('language', 'de')
        if language == 'de':
            graph = graph_de
            error_response = {"message": {"answer": "Diese Anfrage hat nicht ganz geklappt. Bitte versuchen Sie es erneut."}}

        elif language == 'en':
            graph = graph_en
            error_response = {"message": {"answer": "This query did not quite work out. Please try again."}}


        if mockup_response == "True":
            try:
                # try to load local JSON
                with open("response.json", "r") as f:
                    model_output = json.load(f)
                    logger.info("Local response found and loaded")
                    return JsonResponse({"message": model_output})

            except:
                logger.info("No local response found, querying OpenAI")
                async_respond_to_query = sync_to_async(respond_to_query)
                model_output = await async_respond_to_query(user_input, graph)

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
                async_respond_to_query = sync_to_async(respond_to_query)
                model_output = await async_respond_to_query(user_input, graph)
                # check if answer in needed format
                if isinstance(model_output["answer"], dict):
                    with open("response.json", "w") as f:
                        json.dump(model_output, f, indent=2)
                    return JsonResponse({"message": model_output})
                # else try to reformat the answer
                else:
                    try:
                        dict_cleaned = dict(model_output["answer"].replace("```json\n", "").replace("```", "").strip())
                        model_output["answer"] = dict_cleaned
                        with open("response.json", "w") as f:
                            json.dump(model_output, f, indent=2)
                        return JsonResponse({"message": model_output})
                    except:
                        logger.info('Response from OpenAI not in expected format')
                        return JsonResponse(error_response, status=400)
            except:
                logger.info("No response from OpenAI")
                return JsonResponse(error_response, status=400)

    else:
        # if not a POST request return error
        logger.info("No POST request")
        return JsonResponse(error_response, status=400)