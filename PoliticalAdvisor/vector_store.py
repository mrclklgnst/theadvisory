import os
import logging
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from pinecone import Pinecone as PineconeClient, ServerlessSpec

# Get the logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the embedding model
model_name = "all-MiniLM-L6-v2"  # A good balance between speed and performance
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


def init_pinecone():
    """Initialize Pinecone client"""
    try:
        # Initialize the Pinecone client
        pc = PineconeClient(
            api_key=os.getenv('PINECONE_API_KEY')
        )

        index_name = os.getenv('PINECONE_INDEX_NAME', 'political-advisory')

        # Check if index exists and create if needed
        try:
            indexes = pc.list_indexes()

            if index_name not in indexes:
                pc.create_index(
                    name=index_name,
                    dimension=384,  # all-MiniLM-L6-v2 embeddings are 384 dimensions
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                logger.info(f"Created new Pinecone index: {index_name}")
            else:
                logger.info(f"Using existing Pinecone index: {index_name}")
        except Exception as e:
            logger.warning(f"Error checking/creating index: {e}")
            logger.info(f"Attempting to use existing index: {index_name}")

        # Initialize the vector store
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            text_key="text"
        )

        return vector_store
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}")
        raise


def add_documents(vector_store: PineconeVectorStore, documents: List[Document]):
    """Add documents to the vector store"""
    try:
        vector_store.add_documents(documents)
        logger.info(
            f"Successfully added {len(documents)} documents to Pinecone")
    except Exception as e:
        logger.error(f"Error adding documents to Pinecone: {e}")
        raise


def similarity_search(vector_store: PineconeVectorStore, query: str, k: int = 3):
    """Search for similar documents"""
    try:
        results = vector_store.similarity_search_with_score(
            query=query,
            k=k
        )
        return results
    except Exception as e:
        logger.error(f"Error searching Pinecone: {e}")
        raise


def delete_all(vector_store: PineconeVectorStore):
    """Delete all vectors from the index"""
    try:
        vector_store.delete(delete_all=True)
        logger.info("Successfully deleted all vectors from Pinecone")
    except Exception as e:
        logger.error(f"Error deleting vectors from Pinecone: {e}")
        raise
