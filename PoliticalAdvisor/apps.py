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

        from .pinecone_rag import build_vector_store, build_graph, build_graph_en
        from .vector_store import init_pinecone, delete_all

        rebuild_index = os.environ.get("REBUILD_INDEX", default=True)

        # Define party programs
        pdf_list = [
            'AFD_Program.pdf',
            'BSW_Program.pdf',
            'CDU_Program.pdf',
            'FDP_Program.pdf',
            'Gruene_Program.pdf',
            'Linke_Program.pdf',
            'Volt_Program.pdf',
            'SPD_Program.pdf'
        ]
        pdf_list_en = [
            'AFD_Program_en.pdf',
            'BSW_Program_en.pdf',
            'CDU_Program_en.pdf',
            'FDP_Program_en.pdf',
            'Gruene_Program_en.pdf',
            'Linke_Program_en.pdf',
            'Volt_Program_en.pdf',
            'SPD_Program_en.pdf'
        ]

        if rebuild_index == 'True':
            logger.info("Building vector stores")

            # Initialize vector stores first
            temp_vector_store = init_pinecone()

            # Delete all existing vectors before rebuilding
            try:
                logger.info(
                    "Deleting all existing vectors from Pinecone index")
                delete_all(temp_vector_store)
                logger.info(
                    "Successfully deleted all vectors from Pinecone index")
            except Exception as e:
                logger.error(f"Error deleting vectors from Pinecone: {e}")
                logger.info(
                    "Continuing with index rebuild despite deletion error")

            # Now build the vector stores
            self.global_vector_store = build_vector_store(pdf_list)
            logger.info('Built DE vector store')
            self.global_vector_store_en = build_vector_store(pdf_list_en)
            logger.info('Built EN vector store')
        else:
            logger.info('Skipped building of vector stores')
            from .vector_store import init_pinecone
            self.global_vector_store = init_pinecone()
            logger.info('Loaded DE vector store')
            self.global_vector_store_en = init_pinecone()
            logger.info('Loaded EN vector store')

        self.global_graph = build_graph(self.global_vector_store)
        logger.info('Built DE graph')

        self.global_graph_en = build_graph_en(self.global_vector_store_en)
        logger.info('Built EN graph')

        logger.info("Vector stores ready")
        logger.info(f"Ready function called (PID: {os.getpid()})")
