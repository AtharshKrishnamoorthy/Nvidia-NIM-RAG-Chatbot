import streamlit as st
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv


load_dotenv()


st.set_page_config(
    page_title="Nvidia NIM RAG Chatbot",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None
)


DB_FAISS_PATH = 'vectorstore/db_faiss'


if not os.getenv("NVIDIA_API_KEY"):
    st.error("NVIDIA API Key not found in environment variables. Please set NVIDIA_API_KEY.")
    st.stop()

os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

def ensure_directory_exists(path):
    """Ensure the directory exists, create if it doesn't."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def process_pdfs():
    """Process uploaded PDF files and convert them to documents."""
    docs = None
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        docs = []
        for file in uploaded_files:
            try:
                filename = file.name 
                with st.spinner(f"Processing {filename}..."):
                    temp_file_path = os.path.join(os.getcwd(), f"temp_{filename}")
                    
                    try:
            
                        with open(temp_file_path, "wb") as temp_file:
                            temp_file.write(file.getvalue())
                        
                       
                        loader = PyPDFLoader(temp_file_path)
                        docs.extend(loader.load())
                        
                    finally:
                      
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                
            except Exception as e:
                st.error(f"Error processing {getattr(file, 'name', 'file')}: {str(e)}")
                continue
        
        if docs:
            st.sidebar.success(f"Successfully processed {len(uploaded_files)} PDF(s)")
    return docs

def create_vectorstore(docs):
    """Create a vector store from the processed documents."""
    if docs is None:
        return None
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        
        embeddings = NVIDIAEmbeddings()
        vectorstore_db = FAISS.from_documents(split_docs, embeddings)
        
        
        ensure_directory_exists(DB_FAISS_PATH)
        vectorstore_db.save_local(DB_FAISS_PATH)
        
        return vectorstore_db
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def generate_response(input_text, memory):
    """Generate a response using the RAG pipeline."""
    try:
        llm = ChatNVIDIA(model_name="nvidia/llama-3.1-nemotron-70b-instruct")
        
        prompt_template = """
        Answer the question based on the PDF context and conversation history.
        Provide it in a structured way.
        
        Previous conversation:
        {history}
        
        <context>
        {context}
        </context>
        
        Question: {input}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        
        if not os.path.exists(DB_FAISS_PATH):
            raise FileNotFoundError("Vector store not found. Please upload and process documents first.")
            
        embeddings = NVIDIAEmbeddings()
        vectorstore_db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore_db.as_retriever()
        
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({
            "input": input_text,
            "history": memory.load_memory_variables({})["history"]
        })
        answer = response["answer"]
        
        memory.save_context({"input": input_text}, {"output": answer})
        return answer
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        st.error(error_msg)
        return f"I apologize, but I encountered an error: {error_msg}"

def initialize_session_state():
    """Initialize the session state variables."""
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def format_chat_history(history):
    """Format the chat history for display."""
    formatted_history = []
    try:
        memory_vars = history.load_memory_variables({})
        messages = memory_vars.get("history", [])
        
        for message in messages:
            if hasattr(message, 'content') and hasattr(message, 'type'):
                role = "assistant" if message.type == "ai" else "user"
                formatted_history.append({
                    "role": role,
                    "content": message.content
                })
    except Exception as e:
        st.error(f"Error formatting chat history: {str(e)}")
    return formatted_history

def main():
    """Main application function."""
    st.title("PDF RAG Chatbot using Nvidia NIM")
    
    
    initialize_session_state()
    
    
    with st.sidebar:
        st.header("Upload Files")
        docs = process_pdfs()
        if docs:
            with st.spinner("Creating Embeddings and Vector Store..."):
                db = create_vectorstore(docs)
                if db:
                    st.success("Vector store created successfully!")
        
       
        if st.button("Clear Chat History"):
            st.session_state.memory = ConversationBufferMemory(
                memory_key="history",
                return_messages=True
            )
            st.session_state.chat_history = []
            st.rerun()
        
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### About")
        st.sidebar.markdown("RAG Chatbot v1.0")
    
   
    chat_container = st.container()
    
    
    with chat_container:
        
        memory_messages = format_chat_history(st.session_state.memory)
        for msg in memory_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    
   
    if prompt := st.chat_input("Ask a question about your documents..."):
     
        if not os.path.exists(DB_FAISS_PATH):
            st.error("Please upload and process documents before asking questions.")
            return
        
      
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, st.session_state.memory)
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
