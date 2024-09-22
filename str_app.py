import streamlit as st
import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import warnings
import re
import time
import logging

# Load environment variables from .env file
load_dotenv()

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers')
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub')
warnings.filterwarnings("ignore")

# Set device to CPU to avoid out of memory errors
device = "cpu"

# Initialize conversation history in session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

def main():
    st.set_page_config(page_title="Medical Chatbot", page_icon="🩺", layout="wide")

    @st.cache_resource
    def download_hugging_face_embeddings():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    embeddings = download_hugging_face_embeddings()

    @st.cache_resource
    def load_pdf_documents(_data_folder="data"):
        all_documents = []
        for filename in os.listdir(_data_folder):
            if filename.endswith(".pdf"):
                file_path = os.path.join(_data_folder, filename)
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                all_documents.extend(documents)
        return all_documents

    documents = load_pdf_documents()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    @st.cache_resource
    def create_faiss_index(_embedding_model, _documents):
        texts = [doc.page_content for doc in _documents]
        return FAISS.from_texts(texts, _embedding_model)

    faiss_index = create_faiss_index(embeddings, text_chunks)

    logging.basicConfig(level=logging.INFO)

    @st.cache_resource
    def load_model():
        print("Starting to load the model...")
        start_time = time.time()
        model_directory = "D:\\inotequia\\med2\\TinyLlama"  
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_directory)
            model = AutoModelForCausalLM.from_pretrained(model_directory)
            model.to(device)
            logging.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
            return tokenizer, model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise e

    tokenizer, model = load_model()

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=500,
        temperature=0.6,
        max_new_tokens=100,
        do_sample=True,
        device=device
    )

    llm_pipeline_instance = HuggingFacePipeline(pipeline=llm_pipeline)

    def retrieve_context(question):
        try:
            relevant_docs = faiss_index.similarity_search(question, k=3)
            context = " ".join([doc.page_content for doc in relevant_docs])
            return context[:800]
        except Exception as e:
            logging.error(f"Error retrieving context: {e}")
            return ""

    def generate_prompt(question, context):
        return f"""
        You are a knowledgeable and empathetic medical assistant. Provide a clear, accurate, and supportive response based on the context provided.

        User's Question: "{question}"

        Context Information: "{context}"

        Instructions:
        - Be concise and relevant.
        - Maintain a compassionate and gentle tone.
        - Ensure the response is comprehensive and informative.

        Response:
        """

    def custom_qa_chain(question):
        try:
            context = retrieve_context(question)
            prompt = generate_prompt(question, context)
            
            result = llm_pipeline_instance(prompt)
            
            if isinstance(result, str):
                # Clean up any stray HTML tags
                response = re.sub(r'</?\w+[^>]*>', '', result)
                match = re.search(r'Response:\s*(.*)', response, re.DOTALL)
                if match:
                    final_response = match.group(1).strip()
                    if len(final_response) < 1000:
                        final_response += " Feel free to ask if you need any further clarification or additional information!"
                    return final_response
            return "I'm sorry, I couldn't find relevant information."
        except Exception as e:
            logging.error(f"Error in custom_qa_chain: {e}")
            return "I'm sorry, something went wrong while processing your request."

    def handle_greetings(user_input):
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        if any(greeting in user_input.lower() for greeting in greetings):
            return ("Hello! 😊 How can I assist you with your medical questions today? "
                    "Feel free to ask about symptoms, treatments, or any general advice.")
        return None

    # CSS styling to make the interface use the full width and remove the white rectangle issue
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f0f2f6;
            width: 100%;
            margin: 0;
            padding: 0;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            padding: 20px;
            width: 100%;
            margin: 0 auto;
        }
        .chat-bubble {
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
            max-width: 75%;
            word-wrap: break-word;
            font-size: 16px;
        }
        .user-bubble {
            background-color: #e0f7fa;
            align-self: flex-end;
            color: #00796b;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .bot-bubble {
            background-color: #fbe9e7;
            align-self: flex-start;
            color: #d32f2f;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .input-container {
            display: flex;
            justify-content: center;
            width: 100%;
            margin: 20px auto;
        }
        .input-box {
            flex-grow: 1;
            margin-right: 10px;
            padding: 12px;
            font-size: 16px;
            border-radius: 20px;
            border: 1px solid #ccc;
        }
        .submit-button {
            padding: 12px;
            font-size: 16px;
            background-color: #00796b;
            color: white;
            border: none;
            cursor: pointer;
            border-radius: 20px;
        }
        .submit-button:hover {
            background-color: #004d40;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Medical Chatbot 🩺")
    st.write("Ask me anything about your medical concerns and I'll assist you!")

    # Display the conversation so far
    st.write("### Chat History")
    with st.container():
        for i, (message, is_user) in enumerate(st.session_state.conversation):
            if is_user:
                st.markdown(f'<div class="chat-bubble user-bubble">{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble bot-bubble">{message}</div>', unsafe_allow_html=True)

    # Input field for new messages
    st.write("### Type Your Message")
    with st.container():
        user_input = st.text_input("You:", key="user_input")

    # Submission button
    if st.button("Send", key="submit"):
        if user_input:
            st.session_state.conversation.append((user_input, True))

            greeting_response = handle_greetings(user_input)
            if greeting_response:
                bot_response = greeting_response
            else:
                bot_response = custom_qa_chain(user_input)

            st.session_state.conversation.append((bot_response, False))

if __name__ == "__main__":
    main()
