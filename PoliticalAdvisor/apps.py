
from django.apps import AppConfig
import logging
import os

logger = logging.getLogger(__name__)


class PoliticaladvisorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "PoliticalAdvisor"

    def ready(self):
        from .myFAISS import build_faiss_programs, load_faiss, build_graph
        # Print the current working directory for debugging
        current_directory = os.path.dirname(__file__)
        faiss_path = os.path.join(current_directory, "faiss_index")

        build_faiss_programs(faiss_path)

        global vector_store, graph
        vector_store = load_faiss(faiss_path)
        graph = build_graph(vector_store)
        logger.info("FAISS index loaded")

