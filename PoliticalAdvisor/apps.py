
from django.apps import AppConfig
import logging
import os
import dotenv
dotenv.load_dotenv()

logger = logging.getLogger(__name__)


class PoliticaladvisorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "PoliticalAdvisor"

    def ready(self):
        from .myFAISS import build_faiss_programs, load_faiss, build_graph
        # Print the current working directory for debugging
        current_directory = os.path.dirname(__file__)
        faiss_path = os.path.join(current_directory, "faiss_index")

        rebuild_faiss_index = os.environ.get("REBUILD_FAISS_INDEX", default=True)
        if rebuild_faiss_index == 'True':
            build_faiss_programs(faiss_path)
        else:
            pass

        global vector_store, graph
        vector_store = load_faiss(faiss_path)
        graph = build_graph(vector_store)
        logger.info("FAISS index loaded")

