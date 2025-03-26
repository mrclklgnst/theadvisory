import os
import logging
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain.schema import HumanMessage
from PoliticalAdvisor.vector_store import init_pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def test_mistral():
    """Test Mistral API connection"""
    try:
        llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.7,
            max_tokens=4096,
            api_key=os.getenv('MISTRAL_API_KEY')
        )

        # Test with a simple query
        response = llm.invoke([HumanMessage(content="Hello, how are you?")])
        logger.info(f"Mistral response: {response.content}")
        return True
    except Exception as e:
        logger.error(f"Error testing Mistral: {e}")
        return False


def test_pinecone():
    """Test Pinecone connection"""
    try:
        vector_store = init_pinecone()
        logger.info(
            f"Successfully connected to Pinecone index: {os.getenv('PINECONE_INDEX_NAME', 'political-advisory')}")
        return True
    except Exception as e:
        logger.error(f"Error connecting to Pinecone: {e}")
        return False


if __name__ == "__main__":
    logger.info("Testing Mistral and Pinecone integration...")

    mistral_success = test_mistral()
    pinecone_success = test_pinecone()

    if mistral_success and pinecone_success:
        logger.info("✅ Both Mistral and Pinecone are working correctly!")
    elif mistral_success:
        logger.info("✅ Mistral is working, but ❌ Pinecone has issues.")
    elif pinecone_success:
        logger.info("✅ Pinecone is working, but ❌ Mistral has issues.")
    else:
        logger.info("❌ Both Mistral and Pinecone have issues.")
