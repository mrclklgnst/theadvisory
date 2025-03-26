import os
import logging
import gc
import torch
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import numpy as np

# Get the logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the embedding model
# Using a smaller model that's more memory efficient
model_name = "all-MiniLM-L6-v2"  # This model produces 384-dimensional embeddings
embeddings_model = SentenceTransformer(model_name)

# Create a custom embeddings class that pads vectors to 1024 dimensions


class CustomEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        # Process in smaller batches to save memory
        batch_size = 16  # Smaller batch size
        batches = [texts[i:i + batch_size]
                   for i in range(0, len(texts), batch_size)]
        all_embeddings = []

        for i, batch in enumerate(batches):
            logger.info(
                f"Processing batch {i+1}/{len(batches)} with {len(batch)} documents")

            # Get the original embeddings
            original_embeddings = self.model.encode(
                batch, normalize_embeddings=True)

            # Check the dimensions
            original_dim = original_embeddings.shape[1] if len(
                original_embeddings.shape) > 1 else len(original_embeddings)

            # If not 1024, pad to 1024
            if original_dim != 1024:
                logger.info(
                    f"Padding embeddings from {original_dim} to 1024 dimensions")
                padded_embeddings = []
                for emb in original_embeddings:
                    # Pad with zeros to reach 1024 dimensions
                    padded = np.pad(emb, (0, 1024 - original_dim), 'constant')
                    # Normalize again after padding
                    padded = padded / np.linalg.norm(padded)
                    padded_embeddings.append(padded)
                all_embeddings.extend(padded_embeddings)
            else:
                all_embeddings.extend(original_embeddings.tolist())

            # Force garbage collection to free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_embeddings

    def embed_query(self, text):
        # Get the original embedding
        original_embedding = self.model.encode(text, normalize_embeddings=True)

        # Check the dimensions
        original_dim = len(original_embedding)

        # If not 1024, pad to 1024
        if original_dim != 1024:
            # Pad with zeros to reach 1024 dimensions
            padded = np.pad(original_embedding,
                            (0, 1024 - original_dim), 'constant')
            # Normalize again after padding
            padded = padded / np.linalg.norm(padded)
            return padded.tolist()

        return original_embedding.tolist()


# Create the custom embeddings instance
embeddings = CustomEmbeddings(embeddings_model)

# Log the embedding dimensions
logger.info(f"Using embedding model: {model_name}")
logger.info(f"Original embedding dimensions: 384, padded to 1024")


def init_pinecone():
    """Initialize Pinecone client"""
    try:
        # Initialize the Pinecone client
        pc = PineconeClient(
            api_key=os.getenv('PINECONE_API_KEY')
        )

        index_name = os.getenv('PINECONE_INDEX_NAME', 'advisory')
        logger.info(f"Using Pinecone index: {index_name}")

        # Check if index exists and create if needed
        try:
            indexes = pc.list_indexes()
            logger.info(f"Available Pinecone indexes: {indexes}")

            if index_name not in indexes:
                logger.info(
                    f"Creating new Pinecone index: {index_name} with dimension 1024")
                pc.create_index(
                    name=index_name,
                    dimension=1024,  # Match the embedding model dimension
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                logger.info(f"Created new Pinecone index: {index_name}")
            else:
                logger.info(f"Using existing Pinecone index: {index_name}")
                # Get index details to check dimensions
                index_details = pc.describe_index(index_name)
                logger.info(f"Index details: {index_details}")
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
        # Process in smaller batches to save memory
        batch_size = 32  # Smaller batch size
        batches = [documents[i:i + batch_size]
                   for i in range(0, len(documents), batch_size)]

        logger.info(
            f"Adding {len(documents)} documents in {len(batches)} batches of size {batch_size}")

        for i, batch in enumerate(batches):
            logger.info(
                f"Processing batch {i+1}/{len(batches)} with {len(batch)} documents")
            vector_store.add_documents(batch)

            # Force garbage collection to free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
