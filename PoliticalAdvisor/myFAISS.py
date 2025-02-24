import json
import os
import re
import faiss
import logging
import requests

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
def load_faiss(faiss_path):
    # Load the index
    vector_store = FAISS.load_local(faiss_path,
                                    OpenAIEmbeddings(),
                                    allow_dangerous_deserialization=True)
    return vector_store
def build_faiss_programs(faiss_path):
    # Build the FAISS index from the party programs and store locally
    vector_store = initialize_faiss()
    programfolder = settings.PDF_STORAGE_URL
    # programfolder = os.path.join(os.path.dirname(__file__), 'programs')
    pdf_list = [
                'AFD_Program.pdf',
                # 'CDU_Program.pdf',
                # 'FDP_Program.pdf',
                # 'Gruene_Program.pdf',
                # 'Linke_Program.pdf',
                # 'Volt_Program.pdf',
                # 'SPD_Program.pdf'
                ]
    for pdf in pdf_list:
        pdf_splits = create_pdf_splits(pdf, programfolder)
        vector_store.add_documents(pdf_splits)
    logger.info(f"Created new FAISS index and added {len(pdf_splits)} documents.")
    vector_store.save_local(faiss_path)
def build_faiss_programs_en(faiss_path):
    # Build the FAISS index from the english party programs and store locally
    vector_store = initialize_faiss()
    # programfolder = os.path.join(os.path.dirname(__file__), 'programs_en')
    programfolder = settings.PDF_STORAGE_URL
    pdf_list = [
                'AFD_Program_en.pdf',
                # 'CDU_Program_en.pdf',
                # 'FDP_Program_en.pdf',
                # 'Gruene_Program_en.pdf',
                # 'Linke_Program_en.pdf',
                # 'Volt_Program_en.pdf',
                # 'SPD_Program_en.pdf'
                ]
    for pdf in pdf_list:
        pdf_splits = create_pdf_splits(pdf, programfolder)
        vector_store.add_documents(pdf_splits)
    logger.info(f"Created new FAISS index in english and added {len(pdf_splits)} documents.")
    vector_store.save_local(faiss_path)
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
      "cdu_csu": {{"agreement": 30, "explanation": "Erklärung", "citations": []}},
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
        results = vector_store.similarity_search_with_score(state['question'], k=10)
        for doc, score in results:
            citation = doc.metadata['source'].split("_")[0] + ": "
            cont = doc.page_content
            cont = cont.replace('\n', ' ')
            pattern = r'(?<=\. )([A-Z][^.]*\.)'
            matches = re.findall(pattern, cont)
            content = " ".join(matches)
            citation = citation + content
            doc.page_content = citation
            doc.metadata['score'] = float(score)
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
      "cdu_csu": {{"agreement": 30, "explanation": "Erklärung", "citations": []}},
      "linke": {{"agreement": 20, "explanation": "Erklärung", "citations": []}},
      "fdp": {{"agreement": 60, "explanation": "Erklärung", "citations": []}},
      "gruene": {{"agreement": 40, "explanation": "Erklärung", "citations": []}},
      "spd": {{"agreement": 80, "explanation": "Erklärung", "citations": []}},
      "volt": {{"agreement": 80, "explanation": "Erklärung", "citations": []}}
    }}

    IMPORTANT: Format your answer only as a JSON without additional characters ."""

    # Create a prompt object from the templates
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = []
        results = vector_store.similarity_search_with_score(state['question'], k=10)
        for doc, score in results:
            citation = doc.metadata['source'].split("_")[0] + ": "
            cont = doc.page_content
            cont = cont.replace('\n', ' ')
            pattern = r'(?<=\. )([A-Z][^.]*\.)'
            matches = re.findall(pattern, cont)
            content = " ".join(matches)
            citation = citation + content
            doc.page_content = citation
            doc.metadata['score'] = float(score)
            retrieved_docs.append(doc)
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        try:
            json_response = json.loads(response.content)  # Convert string to JSON
        except json.JSONDecodeError:
            logger.info("Warning: LLM did not return valid JSON!")
            json_response = {"error": "Invalid JSON response from LLM"}

        return {"answer": json_response}  # Now it's a valid JSON object

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
    response['answer'] = result["answer"]

    # Create a list of citations from the similarity search
    citations = {}
    for i, c in enumerate(result["context"]):
        citation_dict = {}
        citation_dict["score"] = c.metadata["score"]
        citation_dict["source"] = c.metadata["source"]
        citation_dict["location"] = str(c.metadata["page"])+" / "+str(c.metadata["total_pages"])
        citation_dict["content"] = c.page_content
        citations[i] = citation_dict

    # Store the citations in the response
    response['citations'] = citations

    return response
























