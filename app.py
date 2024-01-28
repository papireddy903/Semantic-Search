import streamlit as st
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader

model_name = "paraphrase-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

def semantic_search(query, documents, model):
    # Embed the query and documents
    similarity_scores = []
    sorted_results = []
    query_embedding = model.encode(query, convert_to_tensor=True)
    for document in documents:
        document_embeddings = model.encode(document, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
        similarity_scores.append([similarity_score,document])
    # Sort the documents by similarity score in descending order
    # sorted_results.append([similarity_score,document])
    similarity_scores.sort(key=lambda x: x[0], reverse=True)
    
    return similarity_scores

# Streamlit app
st.title("Semantic Search with Streamlit")

# File uploader for multiple PDFs
uploaded_files = st.file_uploader("Upload multiple PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Read PDFs and extract text
    documents = []
    with st.spinner("Extracting text from PDFs..."):
        for uploaded_file in uploaded_files:
            pdf_reader = PdfReader(uploaded_file)
            pdf_text = ""
            for page_num in range(len(pdf_reader.pages)):
                pdf_text += pdf_reader.pages[page_num].extract_text()
            documents.append(pdf_text)

    query = st.text_input("Enter your query:")

    if query:
        # Perform semantic search
        results = semantic_search(query, documents, model)
        
        # Display the results
        st.subheader("Top Result:")
        st.write(f"Most Similar Document: {results[0][1][1]}")
        st.write(f"Similarity Score: {float(results[0][0]):.4f}")



