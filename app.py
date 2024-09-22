from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline  
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os
import warnings
import torch

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers')
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub')

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Function to download HuggingFace embeddings using a free model
def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  

embeddings = download_hugging_face_embeddings()

# Function to load and extract texts from PDF files in the 'data' folder using PyPDFLoader
def load_pdf_documents(data_folder="data"):
    all_documents = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_folder, filename)
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            all_documents.extend(documents)
    return all_documents

# Load the PDF documents
documents = load_pdf_documents()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

print(f"Loaded {len(documents)} documents.")  # Debugging line

# Create FAISS index using LangChain's FAISS wrapper
def create_faiss_index(embedding_model, documents):
    texts = [doc.page_content for doc in documents]
    return FAISS.from_texts(texts, embedding_model)

# Initialize FAISS index
faiss_index = create_faiss_index(embeddings, text_chunks)

# Custom prompt template for the QA system
custom_prompt_template = """
You are a virtual medical assistant with expertise in providing accurate and empathetic responses. 

Patient's Question:
{question}

Context Information:
{context}

Instructions:
- Provide a precise and evidence-based response.
- Ensure the information is up-to-date and relevant.
- If needed, ask clarifying questions or suggest consulting a healthcare professional for further advice.

Your Response:
"""

# Initialize the PromptTemplate
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Load the TinyLlama model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Update to the correct model name
token = os.getenv("HUGGINGFACE_TOKEN")  # Load token from environment variable

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)

# Update the pipeline configuration to specify the device
device = 0 if torch.cuda.is_available() else -1  # Use 0 for GPU, -1 for CPU

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.8,
    max_new_tokens=50,
    do_sample=True,
    device=device  # Specify the device here
)

# Create a HuggingFacePipeline instance directly
llm_pipeline_instance = HuggingFacePipeline(pipeline=llm_pipeline)

# Set up the RetrievalQA chain with the LLM, prompt, and FAISS retriever
qa = RetrievalQA.from_chain_type(
    llm=llm_pipeline_instance,
    chain_type="stuff",
    retriever=faiss_index.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}  # Ensure prompt includes context
)

@app.route("/")
def index():
    return render_template('chat.html')

import re

@app.route("/get/", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print(f"User Input: {msg}")

    # Check for simple greetings and provide a predefined response
    if msg.lower() in ["hello", "hi", "hey", "greetings"]:
        return jsonify({"response": "Hello! I'm here to help you. What can I assist you with today?"})

    # Prepare the context (could be empty or default)
    context = "No relevant medical history provided."  # Default context

    try:
        # Create a query for the model
        result = qa({"query": msg, "context": context})  # Ensure context is passed
        
        # Extract the response from the result
        response = result.get("result", "Sorry, I couldn't find a good response.")
        
        # Clean up the response
        response = response.strip()
        
        # Remove unwanted introductory and context information
        response = re.sub(r'Patient\'s Question:.*?Context Information:', '', response)
        response = re.sub(r'Context Information:.*?Instructions:', '', response)
        response = re.sub(r'Instructions:.*?Your Response:', '', response)
        response = re.sub(r'Your Response:', '', response)
        
        # Clean specific characters
        response = re.sub(r'\\[^\s]*', '', response)  # Remove all symbols starting with \
        response = re.sub(r'\\u[0-9a-fA-F]{4}', '', response)  # Remove Unicode escape sequences
        response = response.replace("\u2014", "—")  # Replace em dash with proper dash
        response = response.replace("\u201c", '"')  # Replace left double quotation mark
        response = response.replace("\u201d", '"')  # Replace right double quotation mark
        
        # Replace newlines with spaces and remove extra spaces
        response = response.replace("\n", " ")  
        response = re.sub(r'\s+', ' ', response)  # Replace multiple spaces with a single space
        
        # Ensure the response is a coherent paragraph
        response = response.strip()  # Final trim to remove leading/trailing spaces

        # Return the cleaned response
        return jsonify({"response": response})  # Return the final response
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "An error occurred while processing your request."}), 500


    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)