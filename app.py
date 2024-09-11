from dotenv import load_dotenv
from docx import Document
from form import book_appointment, user_detail_form
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from PyPDF2 import PdfReader
import re
import streamlit as st


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF files
def extract_pdf_content(pdf_files):
    all_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            all_text += page.extract_text()
    return all_text

# Function to extract text from .docx files
def extract_docx_content(docx_files):
    document_text = ""
    for docx in docx_files:
        doc = Document(docx)
        for paragraph in doc.paragraphs:
            document_text += paragraph.text + "\n"
    return document_text

# Function to extract text from plain .txt files
def extract_txt_content(txt_files):
    text_content = ""
    for txt in txt_files:
        text_content += txt.read().decode("utf-8") + "\n"  # Decoding for byte format
    return text_content

# Function to break the text into smaller chunks
def split_text_segments(full_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    segments = splitter.split_text(full_text)
    return segments

# Function to create and store vector index
def build_vector_index(text_segments):
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.from_texts(text_segments, embedding=embedding_model)
    vector_db.save_local("custom_faiss_index")


# Function to process uploaded files based on file type
def process_uploaded_files(files_uploaded):
    combined_text = ""
    for file in files_uploaded:
        if file.name.endswith('.pdf'):
            combined_text += extract_pdf_content([file])
        elif file.name.endswith('.docx'):
            combined_text += extract_docx_content([file])
        elif file.name.endswith('.txt'):
            combined_text += extract_txt_content([file])
        else:
            st.error(f"Unsupported file format: {file.name}")
    return combined_text


# Function to handle user queries
def handle_user_query(user_query):
    if re.search(r'\bbook\b|\bappointment\b|\bmeeting\b', user_query, re.IGNORECASE):
        book_appointment(user_query)
        return  # Exit to avoid further processing
    elif re.search(r'\bcall me\b|\bcontact me\b', user_query, re.IGNORECASE):
        st.write("It seems like you'd like us to call you. Please provide your details below:")
       
        name, phone, email = user_detail_form()

        if name and phone and email:
            st.write(f"Thank you {name}, We will contact you shortly at {phone} or via email at {email}.")
        return
    else:
        # Check if FAISS index exists
        if not os.path.exists(f"custom_faiss_index/index.faiss"):
            st.error("Please upload your documents and process them first.")
            return
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_db = FAISS.load_local("custom_faiss_index", embedding_model, allow_dangerous_deserialization=True)
        relevant_docs = vector_db.similarity_search(user_query)

        conversational_chain = build_conversational_chain()
        response = conversational_chain(
            {"input_documents": relevant_docs, "query": user_query},
            return_only_outputs=True
        )
        st.write("Response: ", response["output_text"])

# Function to generate the conversational chain
def build_conversational_chain():
    qa_prompt_template = """Please provide a thorough and precise answer based on the given context. If the answer is not present in the context, respond with "The answer cannot be found in the provided context." Avoid making assumptions or providing wrong information.\n\n
    Context:\n {context}\n
    Query: \n{query}\n

    Response:
    """
    ai_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    custom_prompt = PromptTemplate(template=qa_prompt_template, input_variables=["context", "query"])
    conversational_chain = load_qa_chain(ai_model, chain_type="stuff", prompt=custom_prompt)
    return conversational_chain

# Main function to run the Streamlit app
def main():

    st.set_page_config(page_title="Document Chatbot", page_icon="💬", layout="wide")

    # Sidebar layout for file upload and processing
    with st.sidebar:
        st.header("📁 File Upload")
        st.markdown("Upload your document files below. Supported formats: **PDF, DOCX, and TXT**.")
        uploaded_docs = st.file_uploader("Choose your document files:", 
                                          accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
        
        if st.button("🚀 Submit & Process"):
            if uploaded_docs:
                with st.spinner("Processing your documents..."):
                    combined_text = process_uploaded_files(uploaded_docs)
                    text_segments = split_text_segments(combined_text)
                    build_vector_index(text_segments)
                    st.sidebar.success("Files processed successfully! ✅")
            else:
                st.sidebar.error("Please upload at least one document file to proceed.")

        st.markdown("---")
        st.write("📋 Supported Formats: PDF, DOCX, TXT.")
        st.write("📄 Multiple documents supported.")

    # Main page layout
    st.title("🤖 Chatbot - Chat With Documents")
    st.markdown("**Upload your files** and ask any questions about their content.")
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])  # Create columns for centering
    with col2:
        user_query = st.text_input("❓ Ask question from the document files or book an appointment or ask to call user:")

        if user_query:
            with st.spinner("Searching..."):
                handle_user_query(user_query)

if __name__ == "__main__":
    main()
