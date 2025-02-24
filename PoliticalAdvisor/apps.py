
from django.apps import AppConfig
import logging
import os
import dotenv

# get env variables
dotenv.load_dotenv()

# initiate logger
logger = logging.getLogger(__name__)


class PoliticaladvisorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "PoliticalAdvisor"

    def ready(self):
        from .myFAISS import build_faiss_programs, load_faiss, build_graph
        from .myFAISS import build_faiss_programs_en, build_graph_en

        # Get the current working directory for debugging
        current_directory = os.path.dirname(__file__)
        faiss_path = os.path.join(current_directory, "faiss_index")
        faiss_path_en = os.path.join(current_directory, "faiss_index_en")

        rebuild_faiss_index = os.environ.get("REBUILD_FAISS_INDEX", default=True)
        if rebuild_faiss_index == 'True':
            logger.info("Rebuilding FAISS indexes")
            build_faiss_programs(faiss_path)
            build_faiss_programs_en(faiss_path_en)
        else:
            logger.info('Skipped building of FAISS index')


        global global_vector_store, global_graph
        global global_vector_store_en, global_graph_en

        # Load the vector stores into memory
        global_vector_store = load_faiss(faiss_path)
        logger.info('Loaded DE vector store in memory')

        global_graph = build_graph(global_vector_store)
        logger.info('Loaded DE graph in memory')

        global_vector_store_en = load_faiss(faiss_path_en)
        global_graph_en = build_graph_en(global_vector_store_en)

        logger.info("FAISS index loaded")

        logger.info(f"FAISS Ready function called (PID: {os.getpid()})")

