
from django.apps import AppConfig
from django.conf import settings
import logging
import os
import dotenv
import time

# get env variables
dotenv.load_dotenv()

# initiate logger
logger = logging.getLogger(__name__)


class PoliticaladvisorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "PoliticalAdvisor"

    def ready(self):
        if hasattr(self, "initialized"):  # Prevent multiple executions
            return
        self.initialized = True

        from .myFAISS import build_faiss_programs, load_faiss, build_graph
        from .myFAISS import build_graph_en

        # Get the current working directory for debugging
        # Rework to only include local directory indexes
        # In terms of logic, we need two directories locally
        # We need to mimick those in spaces as well, but then also know the individual file names
        faiss_dir_de = 'faiss_index'
        faiss_dir_en = 'faiss_index_en'

        faiss_dir_path_de = os.path.join(settings.VECTOR_STORAGES_URL, faiss_dir_de)
        faiss_dir_path_en = os.path.join(settings.VECTOR_STORAGES_URL, faiss_dir_en)

        rebuild_faiss_index = os.environ.get("REBUILD_FAISS_INDEX", default=True)

        # Define party programs
        pdf_list = [
            'AFD_Program.pdf',
            'CDU_Program.pdf',
            'FDP_Program.pdf',
            'Gruene_Program.pdf',
            'Linke_Program.pdf',
            'Volt_Program.pdf',
            'SPD_Program.pdf'
        ]
        pdf_list_en = [
            'AFD_Program_en.pdf',
            'CDU_Program_en.pdf',
            'FDP_Program_en.pdf',
            'Gruene_Program_en.pdf',
            'Linke_Program_en.pdf',
            'Volt_Program_en.pdf',
            'SPD_Program_en.pdf'
        ]
        if rebuild_faiss_index == 'True':
            logger.info("Rebuilding FAISS indexes")
            build_faiss_programs(faiss_dir_path_de, 'politicaladvisor', faiss_dir_de, pdf_list)
            build_faiss_programs(faiss_dir_path_en, 'politicaladvisor', faiss_dir_en, pdf_list_en)
        else:
            logger.info('Skipped building of FAISS index')

        # Store FAISS as class attributes
        self.global_vector_store = load_faiss(faiss_dir_path_de, 'politicaladvisor', faiss_dir_de)
        logger.info('Loaded DE vector store in memory')

        self.global_graph = build_graph(self.global_vector_store)
        logger.info('Loaded DE graph in memory')

        self.global_vector_store_en = load_faiss(faiss_dir_path_en, 'politicaladvisor', faiss_dir_en)
        logger.info('Loaded EN vector store in memory')

        self.global_graph_en = build_graph_en(self.global_vector_store_en)
        logger.info('Loaded EN graph in memory')

        logger.info("FAISS index loaded")
        logger.info(f"FAISS Ready function called (PID: {os.getpid()})")

