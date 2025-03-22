import streamlit as st
import fitz  # PyMuPDF
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
# import nltk
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from nltk.tokenize import sent_tokenize
import re


# Configure page
st.set_page_config(
    page_title="DeepDive PDF Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {padding: 2rem 3rem;}
    .stButton>button {border-radius: 8px;}
    .stTextInput>div>div>input {border-radius: 8px;}
    .stDownloadButton {margin-top: 20px;}
    div[data-baseweb="input"] {border-radius: 8px;}
    .reportview-container .main .block-container {max-width: 1200px;}
    </style>
    """, unsafe_allow_html=True)

# # Initialize NLTK
# nltk.download('punkt')
# nltk.download('punkt_tab')

# Cache resources
@st.cache_resource
def load_models():
    return {
        'summarizer': LsaSummarizer(),
        'sentence_model': SentenceTransformer('all-MiniLM-L6-v2'),
        'text_generator': pipeline('text-generation', 
                                model=AutoModelForCausalLM.from_pretrained("gpt2"),
                                tokenizer=AutoTokenizer.from_pretrained("gpt2"))
    }

models = load_models()

# PDF Processing Functions
def extract_pdf_data(pdf_file):
    """Extract TOC, text, and page count from PDF"""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        return {
            'toc': doc.get_toc(simple=True),
            'text': "\n".join(page.get_text() for page in doc),
            'pages': doc.page_count
        }

def generate_summary(text, sentences=5):
    """Generate document summary using LSA"""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summary = models['summarizer'](parser.document, sentences)
    return " ".join(str(sentence) for sentence in summary) or "Summary unavailable"

# Search Functions
def create_search_index(text):
    """Create FAISS index for semantic search"""
    paragraphs = [p for p in sent_tokenize(re.sub(r'\s+', ' ', text.strip())) if p]
    embeddings = models['sentence_model'].encode(paragraphs, convert_to_tensor=True).cpu().numpy()
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return index, paragraphs

# UI Components
def sidebar_navigation():
    """Navigation sidebar"""
    st.sidebar.title("🔍 Navigation")
    pages = {
        "🏠 Home": "home",
        "📄 Document Analysis": "analysis"
    }
    return pages[st.sidebar.radio("Go to", list(pages.keys()))]

def display_toc(toc):
    """Display interactive table of contents"""
    if toc:
        with st.expander("📑 Table of Contents", expanded=True):
            for level, title, page in toc:
                st.markdown(f"{' ' * (level-1)}• **{title}** (p{page})")
    else:
        st.warning("No table of contents found in document")

# Main App Logic
current_page = sidebar_navigation()

if current_page == "home":
    st.title("📚 DeepDive PDF Analyzer")
    st.markdown("""
    Welcome to DeepDive RAG App! Upload PDF documents in the Analysis section to:
    - Get automatic summaries
    - Search document content
    - Generate AI-powered insights
    """)

elif current_page == "analysis":
    st.title("📄 Document Analysis")
    
    uploaded_files = st.file_uploader("Upload PDF documents", 
                                    type=["pdf"], 
                                    accept_multiple_files=True,
                                    help="Upload one or more PDF files for analysis")
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            combined_data = {'toc': [], 'text': '', 'pages': 0}
            for file in uploaded_files:
                data = extract_pdf_data(file)
                combined_data['toc'].extend(data['toc'])
                combined_data['text'] += data['text'] + "\n"
                combined_data['pages'] += data['pages']
            
            index, paragraphs = create_search_index(combined_data['text'])
        
        # Document Overview Section
        with st.expander("📌 Document Overview", expanded=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Total Pages", combined_data['pages'])
                st.metric("Processed Text Length", f"{len(combined_data['text']):,} chars")
            with col2:
                st.subheader("Executive Summary")
                st.write(generate_summary(combined_data['text']))
            
        display_toc(combined_data['toc'])

        # Search and Analysis Section
        st.divider()
        query = st.text_input("Ask about the document:", 
                             placeholder="Enter your question here...",
                             help="Ask questions based on the document content")
        
        if query:
            with st.spinner("Analyzing content..."):
                # Semantic Search
                query_embedding = models['sentence_model'].encode([query]).astype('float32')
                distances, indices = index.search(query_embedding, 3)
                context = "\n".join(paragraphs[i] for i in indices[0])
                
                # Generate Response
                response = models['text_generator'](
                    f"Context: {context}\nQuestion: {query}\nAnswer:",
                    max_new_tokens=200,
                    temperature=0.7
                )[0]['generated_text']
            
            st.subheader("AI Analysis")
            st.markdown(f"**Your Question:** {query}")
            st.info(response.split("Answer:")[-1].strip())
            
            with st.expander("View supporting content"):
                for i, idx in enumerate(indices[0], 1):
                    st.markdown(f"{i}. {paragraphs[idx]}")
    else:
        st.info("👆 Upload PDF documents to begin analysis")

# Hide Streamlit default menu
st.markdown("""<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>""", 
            unsafe_allow_html=True)




# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate

# # ✅ Fix: Ensure set_page_config is first
# st.set_page_config(page_title="Chat PDF", page_icon="📚", layout="wide")

# # ✅ Fix: Cache resource
# @st.cache_resource
# def load_models():
#     return {
#         'sentence_model': HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
#         'text_generator': pipeline(
#             'text-generation', 
#             model=AutoModelForCausalLM.from_pretrained("gpt2"),
#             tokenizer=AutoTokenizer.from_pretrained("gpt2"),
#             max_length=1024,  # ✅ Increased input length
#             pad_token_id=50256
#         )
#     }

# models = load_models()

# # ✅ Function to extract text from PDF
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""
#     return text

# # ✅ Split text into chunks for embedding
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     return chunks

# # ✅ Create FAISS vector store from text chunks
# def get_vector_store(text_chunks):
#     embeddings = models['sentence_model']
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# # ✅ Create Langchain conversational chain
# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. 
#     If the answer is not in the context, say "Answer is not available in the context."
    
#     Context:
#     {context}
    
#     Question:
#     {question}
    
#     Answer:
#     """

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(models['text_generator'], chain_type="stuff", prompt=prompt)
#     return chain

# # ✅ Handle user input and generate response
# def user_input(user_question):
#     embeddings = models['sentence_model']
    
#     # ✅ Load FAISS index
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)

#     if docs:
#         context = "\n\n".join([doc.page_content for doc in docs])
        
#         # ✅ Truncate context if too long
#         max_input_length = 1024 - 100  # Keep some space for the response
#         context = context[:max_input_length]

#         # ✅ Generate response
#         chain = get_conversational_chain()
#         response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

#         st.write("### Reply: ", response["output_text"])
#     else:
#         st.write("### Reply: No relevant information found in the document.")

# # ✅ Streamlit UI
# def main():
#     st.header("📚 Interactive RAG-based LLM for Multi-PDF Document Analysis", divider='rainbow')

#     user_question = st.text_input("💬 Ask a question about the PDF files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("📂 Menu:")
#         pdf_docs = st.file_uploader("📥 Upload your PDF files", accept_multiple_files=True)

#         if st.button("🚀 Submit & Process"):
#             if pdf_docs:
#                 with st.spinner("Processing PDFs..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_text_chunks(raw_text)
#                     get_vector_store(text_chunks)
#                     st.success("✅ Documents processed successfully!")
#             else:
#                 st.error("❌ Please upload at least one PDF file.")

# if __name__ == "__main__":
#     main()
