import json
import os
import re
import faiss
import logging
import requests
import boto3

from django.conf import settings

from dotenv import load_dotenv
from typing_extensions import List, TypedDict

from langchain.chat_models import init_chat_model

from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langgraph.graph import START, StateGraph

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Get the logger
logger = logging.getLogger(__name__)

# Initialize the LLM
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Load variables from .env file
load_dotenv()

def initialize_faiss():
    # Instantiate the vector store
    index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query("test"))) # build the index with the dimension of the embeddings
    vector_store = FAISS( # use the index for the vector store
        embedding_function=OpenAIEmbeddings(),
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    return vector_store
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

        loader = PyMuPDFLoader(f"/tmp/{file_path}")  # ✅ Load from local temporary file
    else:
        # Load PDF from local storage
        pdf_path = os.path.join(programfolder, file_path)
        if not os.path.exists(pdf_path):
            raise ValueError(f"ERROR: PDF file not found - {pdf_path}")

        loader = PyMuPDFLoader(pdf_path)
    # split the given pdf given in file path and programfolder into chunks and return

    # Load the pdf PyMUPDFLoader works much better than PyPDFLoader

    pdf_doc = loader.load()
    logger.info(f"Loaded {len(pdf_doc)} pages from {file_path}")

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # split the text into chunks
        chunk_overlap=200,
        add_start_index=True #retains in metadata the where each text split starts
    )
    pdf_splits = text_splitter.split_documents(pdf_doc)
    logger.info(f"Split PDF into {len(pdf_splits)} sub-documents.")

    # Add metadata with source PDF name and a hash of the content to avoid duplicates
    for doc in pdf_splits:
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata['hash'] = hash(doc.page_content)

    if settings.USE_SPACES:
        try:
            os.remove(temp_pdf_path)
            logger.info(f"🗑️ Deleted temporary file {temp_pdf_path}")
        except Exception as e:
            logger.error(f"⚠️ Failed to delete temporary file {temp_pdf_path}: {e}")

    return pdf_splits
def load_faiss(faiss_dir_path, bucket_name, faiss_dir):
    '''Load FAISS index from local file,
    or from DigitalOcean Spaces if not found locally
    :arg faiss_path: Path to the local FAISS index file
    '''
    logger.info('IN load faiss')

    # Ensure `faiss_dir_path` is a directory
    if not os.path.exists(faiss_dir_path):
        logger.info(f"Creating missing FAISS directory: {faiss_dir_path}")
        os.makedirs(faiss_dir_path, exist_ok=True)

    try:
        vector_store = FAISS.load_local(faiss_dir_path,
                                        OpenAIEmbeddings(),
                                        allow_dangerous_deserialization=True)
        logger.info(f"Loaded FAISS index from {faiss_dir_path}")
        return vector_store
    except:
        # Load the FAISS index from DigitalOcean Spaces
        try:
            session = boto3.session.Session()
            endpoint_url = f"https://{os.environ.get('DO_SPACES_ENDPOINT_BARE')}"
            client = session.client(
                's3',
                region_name=os.environ.get('DO_SPACES_REGION'),
                endpoint_url=endpoint_url,
                aws_access_key_id=os.environ.get('DO_SPACES_ACCESS_KEY'),
                aws_secret_access_key=os.environ.get('DO_SPACES_SECRET_KEY')
            )
            logger.info("Connected to DigitalOcean Spaces")
        except:
            logger.error("Failed to connect to DigitalOcean Spaces")
            return None
        try:
            # Download the FAISS index from DigitalOcean Spaces
            logger.info(f"Downloading FAISS index from {bucket_name}")
            client.download_file(bucket_name, 'vector_storages/'+faiss_dir+'/index.faiss', os.path.join(faiss_dir_path, 'index.faiss'))
            client.download_file(bucket_name, 'vector_storages/'+faiss_dir+'/index.pkl', os.path.join(faiss_dir_path, 'index.pkl'))
            logger.info(f"Downloaded FAISS index from {bucket_name}")
        finally:
            client.close()
            logger.info("Closed DigitalOcean Spaces client")

        # Load vectore into memory
        vector_store = FAISS.load_local(faiss_dir_path,
                                        OpenAIEmbeddings(),
                                        allow_dangerous_deserialization=True)
        logger.info(f"Loaded FAISS index from {faiss_dir_path}")

        return vector_store

def build_faiss_programs(faiss_dir_path, bucket_name, faiss_dir, pdf_list):
    '''Build the FAISS index from the party programs, store locally and in cloud
    Args:
        faiss_path (str): Path to the local FAISS index file
        bucket_name (str): Name of the bucket in DigitalOcean Spaces
        pdf_list (list): List of party programs to be used for building the index
    '''
    vector_store = initialize_faiss()
    programfolder = settings.PDF_STORAGE_URL

    # Split party progarms into chunks and add to the vector store
    for pdf in pdf_list:
        pdf_splits = create_pdf_splits(pdf, programfolder)
        vector_store.add_documents(pdf_splits)
    logger.info(f"Created new FAISS index and added {len(pdf_splits)} documents.")

    # Save the FAISS index locally
    vector_store.save_local(faiss_dir_path)
    logger.info(f"Saved FAISS index to {faiss_dir_path}")

    # Save the FAISS index and pickle to DigitalOcean Spaces
    save_to_spaces(os.path.join(faiss_dir_path, 'index.faiss'), bucket_name, 'vector_storages/'+faiss_dir+'/index.faiss')
    save_to_spaces(os.path.join(faiss_dir_path, 'index.pkl'), bucket_name, 'vector_storages/'+faiss_dir+'/index.pkl')

def save_to_spaces(local_path, bucket_name, remote_path):
    """Uploads FAISS index to DigitalOcean Spaces.
    Args:
        local_path (str): Path to the local FAISS index file.
        remote_path (str): Path to the remote quasi folder and file name in DO Spaces
        bucket_name (str): Name of the bucket in DO Spaces
    """
    session = boto3.session.Session()
    endpoint_url = f"https://{os.environ.get('DO_SPACES_ENDPOINT_BARE')}"
    logger.info(endpoint_url)
    client = session.client(
        's3',
        region_name=os.environ.get('DO_SPACES_REGION'),
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ.get('DO_SPACES_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('DO_SPACES_SECRET_KEY')
    )
    logger.info("Connected to DigitalOcean Spaces")

    try:
        # THIS IS WHERE IT FAILS
        client.upload_file(local_path, bucket_name, remote_path)
        logger.info(f"Uploaded FAISS index to {remote_path} on DigitalOcean Spaces")
    finally:
        client.close()
        logger.info("Closed DigitalOcean Spaces client")
def query_faiss(query, vector_store):
    # results = vector_store.similarity_search(query=query, k=3)
    results = vector_store.similarity_search_with_score(query=query, k=3)
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
    Du bist ein Experte für politische Analyse. 
    Analysiere die folgende politische Aussage und gib die Position jeder deutschen Partei zurück.
    Verwende ausschließlich die Informationen im gegebenen Kontext, um deine Antwort zu formulieren.
    Nimm keine weiteren Informationen in deine Antwort auf.
    Kontext: {context}
    Aussage: {question}
    Antworte NUR mit einem JSON-Objekt in diesem Format:
    {{
      "afd": {{"agreement": 75, "explanation": "Erklärung", "citations": []}},
      "bsw": {{"agreement": 50, "explanation": "Erklärung", "citations": []}},
      "cdu": {{"agreement": 30, "explanation": "Erklärung", "citations": []}},
      "linke": {{"agreement": 20, "explanation": "Erklärung", "citations": []}},
      "fdp": {{"agreement": 60, "explanation": "Erklärung", "citations": []}},
      "gruene": {{"agreement": 40, "explanation": "Erklärung", "citations": []}},
      "spd": {{"agreement": 80, "explanation": "Erklärung", "citations": []}},
      "volt": {{"agreement": 80, "explanation": "Erklärung", "citations": []}}
    }}

    WICHTIG: Formatiere die Antwort NUR als valides JSON ohne zusätzlichen Text oder Zeichen."""

    # Create a prompt object from the templates
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Define application steps
    def retrieve(state: State):
        # Pull relevant docs from the vector stored
        retrieved_docs = []
        results = vector_store.similarity_search_with_score(state['question'], k=30)
        for doc, score in results:
            party = doc.metadata['source'].split("_")[0].lower()
            doc.metadata['party'] = party
            citation = doc.metadata['source'].split("_")[0] + ": "
            cont = doc.page_content
            cont = cont.replace('\n', ' ')
            pattern = r'(?<=\. )([A-Z][^.]*\.)'
            matches = re.findall(pattern, cont)
            content = " ".join(matches)
            citation = citation + content
            doc.page_content = citation
            doc.metadata['score'] = float(score)
            if len(citation)>100:
                retrieved_docs.append(doc)
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        # turn response into a json
        try:
            json_response = json.loads(response.content)
        except json.JSONDecodeError:
            logger.info("Warning: LLM did not return valid JSON!")
            logger.info(f"LLM response: {response.content}")
            try:
                logger.info("Trying to clean up the response")
                json_response = json.loads(response.content.replace("```json\n", "").replace("```", "").strip())
                logger.info("Cleaned up the response")
            except:
                logger.info("Failed to clean up the response")
                json_response = {"error": "Invalid JSON response from LLM"}

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
    You are a political analyst
    Analyse the following political statement and return the position of each German party.
    Exclusively use the information in the given context to formulate your answer.
    Do not include any additional information in your answer.
    Context: {context}
    Statement: {question}
    Answer ONLY with a JSON object in this format:
    {{
      "afd": {{"agreement": 75, "explanation": "Erklärung", "citations": []}},
      "bsw": {{"agreement": 50, "explanation": "Erklärung", "citations": []}},
      "cdu": {{"agreement": 30, "explanation": "Erklärung", "citations": []}},
      "linke": {{"agreement": 20, "explanation": "Erklärung", "citations": []}},
      "fdp": {{"agreement": 60, "explanation": "Erklärung", "citations": []}},
      "gruene": {{"agreement": 40, "explanation": "Erklärung", "citations": []}},
      "spd": {{"agreement": 80, "explanation": "Erklärung", "citations": []}},
      "volt": {{"agreement": 80, "explanation": "Erklärung", "citations": []}}
    }}
    IMPORTANT: Answer in english.
    IMPORTANT: Format your answer only as a JSON without additional characters."""


    # Create a prompt object from the templates
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = []
        results = vector_store.similarity_search_with_score(state['question'], k=30)
        for doc, score in results:
            party = doc.metadata['source'].split("_")[0].lower()
            doc.metadata['party'] = party
            citation = doc.metadata['source'].split("_")[0] + ": "
            cont = doc.page_content
            cont = cont.replace('\n', ' ')
            pattern = r'(?<=\. )([A-Z][^.]*\.)'
            matches = re.findall(pattern, cont)
            content = " ".join(matches)
            citation = citation + content
            doc.page_content = citation
            doc.metadata['score'] = float(score)
            if len(citation)>100:
                retrieved_docs.append(doc)
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        try:
            json_response = json.loads(response.content)
        except json.JSONDecodeError:
            logger.info("Warning: LLM did not return valid JSON!")
            logger.info(f"LLM response: {response.content}")
            try:
                logger.info("Trying to clean up the response")
                json_response = json.loads(response.content.replace("```json\n", "").replace("```", "").strip())
                logger.info("Cleaned up the response")
            except:
                logger.info("Failed to clean up the response")
                json_response = {"error": "Invalid JSON response from LLM"}

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
                dict_cleaned = dict(answer_raw.replace("```json\n", "").replace("```", "").strip())
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
                citation_dict["location"] = str(c.metadata["page"])+" / "+str(c.metadata["total_pages"])
                citation_dict["content"] = c.page_content
                citations_structured[party].append(citation_dict)
                party_citation_count += 1
        response["answer"][party]["citations_count"] = party_citation_count

    # Add list of citations to party answer
    for party in party_list:
        response["answer"][party]["citations"] = citations_structured[party]

    return response
























