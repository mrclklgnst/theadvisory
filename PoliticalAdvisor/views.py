from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from asgiref.sync import sync_to_async
from .myFAISS import respond_to_query
from django.apps import apps
import json
import dotenv
import os
import logging
import random

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
            'quick_topics_intro': "If you're not sure what to ask, click on one of the suggestions for topics:",
            'quick_messages_intro': "Below are some questions you can ask about the topic you selected:",
            'bot_init_message': "Hello! I am a political advisor bot. Below you can enter a political statement that is important to you and I will tell you which political parties have strong positions relevant to your statement. If you want to see the sources of the information, click on the 'Citations' button. If you want to change the language, click on the language selector top.",
        },
        'de': {
            'message_prompt': 'Geben Sie eine politische Aussage ein die Ihnen wichtig ist ....',
            'button_text': 'Versenden',
            'table_title': 'Finden Sie unten eine KI generierte Zusammenfassung der Parteistandpunkte und relevante Zitate aus den Partei Programmen',
            'citations_title': 'Zitate aus Parteiprogrammen',
            'quick_topics_intro': "Wenn Sie nicht sicher sind, was Sie fragen sollen, klicken Sie auf eine der Themen-Vorschläge:",
            'quick_messages_intro': "Hier sind einige Fragen, die Sie zu dem von Ihnen ausgewählten Thema stellen können:",
            'bot_init_message': "Hallo! Ich bin ein politischer Berater-Bot. Hier können Sie eine politische Aussage eingeben, die Ihnen wichtig ist, und ich werde Ihnen sagen, welche politischen Parteien starke Positionen haben, die für Ihre Aussage relevant sind. Wenn Sie die Quellen der Informationen sehen möchten, klicken Sie auf die Schaltfläche 'Zitate'. Wenn Sie die Sprache ändern möchten, klicken Sie auf den Sprachauswahl oben.",
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
        language = request.COOKIES.get('language')
        logger.info(f"Receiveed user input: {user_input} in language: {language}")
        if language == 'de':
            graph = graph_de
            error_response = {"message": {"answer": "Diese Anfrage hat nicht ganz geklappt. Bitte versuchen Sie es erneut."}}

        elif language == 'en':
            graph = graph_en
            error_response = {"message": {"answer": "This query did not quite work out. Please try again."}}

        # Get suggested prompts
        suggested_prompts = createRandomPrompts(language)
        error_response["suggested_prompts"] = suggested_prompts

        if mockup_response == "True":
            try:
                # try to load local JSON
                with open("response.json", "r") as f:
                    model_output = json.load(f)
                    logger.info("Local response found and loaded")
                    return JsonResponse({"message": model_output, "suggested_prompts": suggested_prompts})

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
                        return JsonResponse({"message": model_output, "suggested_prompts": suggested_prompts})
        else:
            try:
                # query openAI, returns dict with keys 'answer' and 'citations'
                async_respond_to_query = sync_to_async(respond_to_query)
                model_output = await async_respond_to_query(user_input, graph)
                # check if answer in needed format
                if isinstance(model_output["answer"], dict):
                    with open("response.json", "w") as f:
                        json.dump(model_output, f, indent=2)
                    return JsonResponse({"message": model_output, "suggested_prompts": suggested_prompts})
                # else try to reformat the answer
                else:
                    try:
                        dict_cleaned = dict(model_output["answer"].replace("```json\n", "").replace("```", "").strip())
                        model_output["answer"] = dict_cleaned
                        with open("response.json", "w") as f:
                            json.dump(model_output, f, indent=2)
                        return JsonResponse({"message": model_output, "suggested_prompts": suggested_prompts})
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

def createRandomPrompts(language):
    '''
    This function creates a dictionary with 3 random categories and 3 random statements from each category
    :param language:
    :return:
    '''
    with open('PoliticalAdvisor/statements_enriched.json', 'r') as f:
        statement_dict = json.load(f)
    statement_dict = statement_dict[language]

    resp_dict = {}

    # Selecting 3 random categories from the dictionary
    random_keys = random.sample(list(statement_dict), 3)

    # Selecting 3 random statements from the selected categories
    for k in random_keys:
        resp_dict[k] = random.sample(statement_dict[k], 2)

    return resp_dict

def create_init_prompts(request):
    '''
    This function creates a dictionary with 3 random categories and 3 random statements from each category
    :param language:
    :return:
    '''

    if request.method == "POST":
        language = request.COOKIES.get('language')
    with open('PoliticalAdvisor/statements_enriched.json', 'r') as f:
        statement_dict = json.load(f)
    statement_dict = statement_dict[language]

    resp_dict = {}

    # Selecting 3 random categories from the dictionary
    random_keys = random.sample(list(statement_dict), 3)

    # Selecting 3 random statements from the selected categories
    for k in random_keys:
        resp_dict[k] = random.sample(statement_dict[k], 2)

    logger.info(f"Initial prompts: {resp_dict}")

    return JsonResponse(resp_dict)
