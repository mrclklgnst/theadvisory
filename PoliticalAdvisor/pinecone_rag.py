import json
import os
import re
import logging
import requests
import boto3

from django.conf import settings
from dotenv import load_dotenv
from typing_extensions import List, TypedDict

from langchain_mistralai import ChatMistralAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

from .vector_store import init_pinecone, add_documents, similarity_search

# Get the logger
logger = logging.getLogger(__name__)

# Initialize the LLM
llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv('MISTRAL_API_KEY')
)

# Load variables from .env file
load_dotenv()


def initialize_vector_store():
    """Initialize Pinecone vector store"""
    return init_pinecone()


def create_pdf_splits(file_path, programfolder):
    if settings.USE_SPACES:
        pdf_url = f"{settings.PDF_STORAGE_URL}{file_path}"
        response = requests.get(pdf_url)
        if response.status_code != 200:
            raise ValueError(f"ERROR: Unable to download PDF from {pdf_url}")

        # Save PDF temporarily before processing
        temp_pdf_path = f"/tmp/{file_path}"
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)

        loader = PyMuPDFLoader(f"/tmp/{file_path}")
    else:
        # Load PDF from local storage
        pdf_path = os.path.join(programfolder, file_path)
        if not os.path.exists(pdf_path):
            raise ValueError(f"ERROR: PDF file not found - {pdf_path}")

        loader = PyMuPDFLoader(pdf_path)

    pdf_doc = loader.load()
    logger.info(f"Loaded {len(pdf_doc)} pages from {file_path}")

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        add_start_index=True
    )
    pdf_splits = text_splitter.split_documents(pdf_doc)
    logger.info(f"Split PDF into {len(pdf_splits)} sub-documents.")

    # Add metadata with source PDF name and a hash of the content
    for doc in pdf_splits:
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata['hash'] = hash(doc.page_content)

    if settings.USE_SPACES:
        try:
            os.remove(temp_pdf_path)
            logger.info(f"🗑️ Deleted temporary file {temp_pdf_path}")
        except Exception as e:
            logger.error(
                f"⚠️ Failed to delete temporary file {temp_pdf_path}: {e}")

    return pdf_splits


def build_vector_store(pdf_list):
    """Build the vector store from the party programs
    Args:
        pdf_list (list): List of party programs to be used for building the index
    """
    vector_store = initialize_vector_store()
    programfolder = settings.PDF_STORAGE_URL

    # Split party programs into chunks and add to the vector store
    for pdf in pdf_list:
        pdf_splits = create_pdf_splits(pdf, programfolder)
        add_documents(vector_store, pdf_splits)

    logger.info(
        f"Created new vector store and added documents from {len(pdf_list)} PDFs.")
    return vector_store


def query_vector_store(query, vector_store):
    results = similarity_search(vector_store, query, k=3)
    for doc, score in results:
        citation = doc.metadata['source'].split("_")[0] + ": "
        cont = doc.page_content
        cont = cont.replace('\n', ' ')
        pattern = r'(?<=\. )([A-Z][^.]*\.)'
        matches = re.findall(pattern, cont)
        content = " ".join(matches)
        citation = citation + content
        print(f"Score: {score}")
        print(f"{citation[:300]}")
        print(f"{doc.metadata['source']}")
        print("-"*50)


def build_graph(vector_store):
    # Build a graph to be used to process user queries

    # Define prompt template
    template = """
    You are an expert political analyst specializing in German politics.

    TASK:
    Analyze the following political statement and determine the position of each German political party based ONLY on the provided context. Do not use any prior knowledge.

    CONTEXT:
    {context}

    STATEMENT TO ANALYZE:
    {question}

    INSTRUCTIONS:
    1. For each party, provide:
       - An agreement score (0-100) indicating how strongly the party agrees with the statement
       - A brief explanation (1-2 sentences) justifying the score
       - Leave citations empty as they will be filled later

    2. Base your analysis EXCLUSIVELY on the context provided. If the context doesn't mention a party's position on the topic, make a reasonable inference based on related positions in the context.

    3. Be objective and politically neutral in your analysis.

    RESPONSE FORMAT:
    Respond ONLY with a valid JSON object in this exact format:
    {{
      "afd": {{"agreement": <score>, "explanation": "<explanation>", "citations": []}},
      "bsw": {{"agreement": <score>, "explanation": "<explanation>", "citations": []}},
      "cdu": {{"agreement": <score>, "explanation": "<explanation>", "citations": []}},
      "linke": {{"agreement": <score>, "explanation": "<explanation>", "citations": []}},
      "fdp": {{"agreement": <score>, "explanation": "<explanation>", "citations": []}},
      "gruene": {{"agreement": <score>, "explanation": "<explanation>", "citations": []}},
      "spd": {{"agreement": <score>, "explanation": "<explanation>", "citations": []}},
      "volt": {{"agreement": <score>, "explanation": "<explanation>", "citations": []}}
    }}

    IMPORTANT: Your response must be ONLY the JSON object with no additional text, markdown formatting, or explanations.
    """

    # Create a prompt object from the templates
    prompt = PromptTemplate(template=template, input_variables=[
                            "context", "question"])

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Define application steps
    def retrieve(state: State):
        # Pull relevant docs from the vector stored
        retrieved_docs = []
        # Use standard similarity search
        results = vector_store.similarity_search_with_score(
            state['question'], k=15
        )

        # Group documents by party for better representation
        party_docs = {}

        for doc, score in results:
            party = doc.metadata['source'].split("_")[0].lower()
            doc.metadata['party'] = party

            # Improve content extraction
            content = doc.page_content
            content = content.replace('\n', ' ').strip()

            # Extract meaningful sentences (improved pattern)
            pattern = r'([A-Z][^.!?]*[.!?])'
            matches = re.findall(pattern, content)

            # Create a more informative citation
            if matches:
                # Take up to 3 most relevant sentences
                relevant_content = " ".join(matches[:3])
                citation = f"{doc.metadata['source'].split('_')[0]}: {relevant_content}"
            else:
                # Fallback to a portion of the original content if no sentences found
                citation = f"{doc.metadata['source'].split('_')[0]}: {content[:200]}..."

            doc.page_content = citation
            doc.metadata['score'] = float(score)

            # Only add if citation is substantial
            if len(citation) > 100:
                # Group by party to ensure representation from all parties
                if party not in party_docs:
                    party_docs[party] = []
                party_docs[party].append(doc)

        # Take the top documents from each party to ensure balanced representation
        for party, docs in party_docs.items():
            # Sort by score (highest first)
            sorted_docs = sorted(
                docs, key=lambda x: x.metadata['score'], reverse=True)
            # Take up to 2 best documents per party
            retrieved_docs.extend(sorted_docs[:2])

        return {"context": retrieved_docs}

    def generate(state: State):
        # Format context for better readability
        formatted_docs = []
        for doc in state["context"]:
            party = doc.metadata.get('party', 'unknown').upper()
            formatted_docs.append(f"[{party}] {doc.page_content}")

        # Join with clear separators
        docs_content = "\n\n---\n\n".join(formatted_docs)

        # Invoke the LLM with the formatted context
        messages = prompt.invoke({
            "question": state["question"],
            "context": docs_content
        })

        # Set temperature lower for more consistent outputs
        response = llm.invoke(messages, temperature=0.3)

        # Process the response
        try:
            # First try direct JSON parsing
            json_response = json.loads(response.content)
            logger.info("Successfully parsed JSON response")
        except json.JSONDecodeError:
            logger.warning(
                "LLM did not return valid JSON, attempting to clean up")
            logger.debug(f"Raw LLM response: {response.content}")

            # Try to extract JSON from markdown code blocks
            try:
                # Extract content from markdown code blocks
                json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
                match = re.search(json_pattern, response.content)

                if match:
                    json_str = match.group(1).strip()
                    json_response = json.loads(json_str)
                    logger.info(
                        "Successfully extracted and parsed JSON from code block")
                else:
                    # Try to clean up common formatting issues
                    cleaned_content = response.content.replace(
                        "```json\n", "").replace("```", "").strip()
                    json_response = json.loads(cleaned_content)
                    logger.info("Successfully parsed JSON after cleanup")
            except Exception as e:
                logger.error(f"Failed to parse JSON after cleanup: {e}")
                json_response = {
                    "error": "Invalid JSON response from LLM",
                    # Include truncated raw response for debugging
                    "raw_response": response.content[:500]
                }

        # Validate the response structure
        required_parties = ["afd", "bsw", "cdu",
                            "linke", "fdp", "gruene", "spd", "volt"]
        for party in required_parties:
            if party not in json_response:
                json_response[party] = {
                    "agreement": 50,
                    "explanation": f"No clear position found for {party.upper()} in the provided context.",
                    "citations": []
                }

        return {"answer": json_response}

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph


def build_graph_en(vector_store):
    # Build a graph to be used to process user queries
    # Uses an english prompt

    # Define prompt template
    template = """
    You are an expert political analyst specializing in German politics.

    TASK:
    Analyze the following political statement and determine the position of each German political party based ONLY on the provided context. Do not use any prior knowledge.

    CONTEXT:
    {context}

    STATEMENT TO ANALYZE:
    {question}

    INSTRUCTIONS:
    1. For each party, provide:
       - An agreement score (0-100) indicating how strongly the party agrees with the statement
       - A brief explanation (1-2 sentences) justifying the score
       - Leave citations empty as they will be filled later

    2. Base your analysis EXCLUSIVELY on the context provided. If the context doesn't mention a party's position on the topic, make a reasonable inference based on related positions in the context.

    3. Be objective and politically neutral in your analysis.

    RESPONSE FORMAT:
    Respond ONLY with a valid JSON object in this exact format:
    {{
      "afd": {{"agreement": <score>, "explanation": "<explanation>", "citations": []}},
      "bsw": {{"agreement": <score>, "explanation": "<explanation>", "citations": []}},
      "cdu": {{"agreement": <score>, "explanation": "<explanation>", "citations": []}},
      "linke": {{"agreement": <score>, "explanation": "<explanation>", "citations": []}},
      "fdp": {{"agreement": <score>, "explanation": "<explanation>", "citations": []}},
      "gruene": {{"agreement": <score>, "explanation": "<explanation>", "citations": []}},
      "spd": {{"agreement": <score>, "explanation": "<explanation>", "citations": []}},
      "volt": {{"agreement": <score>, "explanation": "<explanation>", "citations": []}}
    }}

    IMPORTANT: Your response must be ONLY the JSON object with no additional text, markdown formatting, or explanations.
    """

    # Create a prompt object from the templates
    prompt = PromptTemplate(template=template, input_variables=[
                            "context", "question"])

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Define application steps
    def retrieve(state: State):
        # Pull relevant docs from the vector stored
        retrieved_docs = []
        # Use standard similarity search
        results = vector_store.similarity_search_with_score(
            state['question'], k=15
        )

        # Group documents by party for better representation
        party_docs = {}

        for doc, score in results:
            party = doc.metadata['source'].split("_")[0].lower()
            doc.metadata['party'] = party

            # Improve content extraction
            content = doc.page_content
            content = content.replace('\n', ' ').strip()

            # Extract meaningful sentences (improved pattern)
            pattern = r'([A-Z][^.!?]*[.!?])'
            matches = re.findall(pattern, content)

            # Create a more informative citation
            if matches:
                # Take up to 3 most relevant sentences
                relevant_content = " ".join(matches[:3])
                citation = f"{doc.metadata['source'].split('_')[0]}: {relevant_content}"
            else:
                # Fallback to a portion of the original content if no sentences found
                citation = f"{doc.metadata['source'].split('_')[0]}: {content[:200]}..."

            doc.page_content = citation
            doc.metadata['score'] = float(score)

            # Only add if citation is substantial
            if len(citation) > 100:
                # Group by party to ensure representation from all parties
                if party not in party_docs:
                    party_docs[party] = []
                party_docs[party].append(doc)

        # Take the top documents from each party to ensure balanced representation
        for party, docs in party_docs.items():
            # Sort by score (highest first)
            sorted_docs = sorted(
                docs, key=lambda x: x.metadata['score'], reverse=True)
            # Take up to 2 best documents per party
            retrieved_docs.extend(sorted_docs[:2])

        return {"context": retrieved_docs}

    def generate(state: State):
        # Format context for better readability
        formatted_docs = []
        for doc in state["context"]:
            party = doc.metadata.get('party', 'unknown').upper()
            formatted_docs.append(f"[{party}] {doc.page_content}")

        # Join with clear separators
        docs_content = "\n\n---\n\n".join(formatted_docs)

        # Invoke the LLM with the formatted context
        messages = prompt.invoke({
            "question": state["question"],
            "context": docs_content
        })

        # Set temperature lower for more consistent outputs
        response = llm.invoke(messages, temperature=0.3)

        # Process the response
        try:
            # First try direct JSON parsing
            json_response = json.loads(response.content)
            logger.info("Successfully parsed JSON response")
        except json.JSONDecodeError:
            logger.warning(
                "LLM did not return valid JSON, attempting to clean up")
            logger.debug(f"Raw LLM response: {response.content}")

            # Try to extract JSON from markdown code blocks
            try:
                # Extract content from markdown code blocks
                json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
                match = re.search(json_pattern, response.content)

                if match:
                    json_str = match.group(1).strip()
                    json_response = json.loads(json_str)
                    logger.info(
                        "Successfully extracted and parsed JSON from code block")
                else:
                    # Try to clean up common formatting issues
                    cleaned_content = response.content.replace(
                        "```json\n", "").replace("```", "").strip()
                    json_response = json.loads(cleaned_content)
                    logger.info("Successfully parsed JSON after cleanup")
            except Exception as e:
                logger.error(f"Failed to parse JSON after cleanup: {e}")
                json_response = {
                    "error": "Invalid JSON response from LLM",
                    # Include truncated raw response for debugging
                    "raw_response": response.content[:500]
                }

        # Validate the response structure
        required_parties = ["afd", "bsw", "cdu",
                            "linke", "fdp", "gruene", "spd", "volt"]
        for party in required_parties:
            if party not in json_response:
                json_response[party] = {
                    "agreement": 50,
                    "explanation": f"No clear position found for {party.upper()} in the provided context.",
                    "citations": []
                }

        return {"answer": json_response}

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph


def respond_to_query(user_query, graph):

    # Invoke the graph with the user query
    result = graph.invoke({"question": user_query})

    # Dictionary with 2 keys: 'answer' and 'citations'
    response = {}

    # Store the chatbot answer
    answer_raw = result["answer"]

    # Clean answer from the chatbot enforcing JSON response
    try:
        # check if answer is a dictionary
        if isinstance(answer_raw, dict):
            response['answer'] = answer_raw
        # else try to reformat the answer
        else:
            try:
                # try cleaning excess characters from the answer
                dict_cleaned = dict(answer_raw.replace(
                    "```json\n", "").replace("```", "").strip())
                response["answer"] = dict_cleaned
            except:
                logger.info('Response from OpenAI not in expected format')
                response["answer"] = answer_raw
    except:
        logger.info('Response from OpenAI not in expected format')
        response["answer"] = answer_raw

    # Create list of parties
    party_list = response['answer'].keys()

    # Sort parties by agreement score
    response['answer'] = {k: v for k, v in sorted(response['answer'].items(),
                                                  key=lambda item: item[1]['agreement'], reverse=True)}

    citations_structured = {}
    for party in party_list:
        citations_structured[party] = []
        party_citation_count = 0
        for i, c in enumerate(result["context"]):
            if c.metadata["party"] == party:
                citation_dict = {}
                citation_dict["score"] = c.metadata["score"]
                citation_dict["source"] = c.metadata["source"]
                citation_dict["location"] = str(
                    c.metadata["page"])+" / "+str(c.metadata["total_pages"])
                citation_dict["content"] = c.page_content
                citations_structured[party].append(citation_dict)
                party_citation_count += 1
        response["answer"][party]["citations_count"] = party_citation_count

    # Add list of citations to party answer
    for party in party_list:
        response["answer"][party]["citations"] = citations_structured[party]

    return response
