# Install necessary packages
!pip install pymupdf python-docx torch pandas sentence-transformers google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client streamlit

import os
import fitz  # PyMuPDF for PDFs
import docx
import torch
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from google.colab import auth
from googleapiclient.discovery import build
from google.colab import drive

# Authenticate and mount Google Drive
auth.authenticate_user()
drive.mount('/content/drive')

# Set up Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive']
SPEECHES_FOLDER = '/content/drive/My Drive/Jim Stone Speeches/Speeches/'
OUTPUT_FOLDER = '/content/drive/My Drive/Jim Stone Speeches/Saved Searches/'

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from files
def extract_text(filepath):
    """Extract text from PDFs and Word documents."""
    text = ""
    if filepath.endswith(".pdf"):
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    elif filepath.endswith(".docx"):
        doc = docx.Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

# Function to load all speeches from Google Drive folder
def load_speeches():
    speeches = []
    for filename in os.listdir(SPEECHES_FOLDER):
        if filename.endswith(".pdf") or filename.endswith(".docx"):
            filepath = os.path.join(SPEECHES_FOLDER, filename)
            text = extract_text(filepath)
            if text:
                embedding = model.encode(text, convert_to_tensor=True).numpy()
                speeches.append({"filename": filename, "content": text, "embedding": embedding})
    return speeches

# Function to search for similar speeches
def search_speeches(query, speeches):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = []
    for speech in speeches:
        stored_embedding = torch.tensor(speech["embedding"])
        similarity = util.pytorch_cos_sim(query_embedding, stored_embedding).item()
        similarities.append((similarity, speech["filename"], speech["content"]))
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:5]

# Function to save search results as a printable file
def save_results(query, results):
    output_filepath = os.path.join(OUTPUT_FOLDER, f"{query.replace(' ', '_')}_results.txt")
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(f"Search Query: {query}\n\n")
        for sim, filename, content in results:
            f.write(f"{filename} (Similarity: {sim:.2f})\n")
            f.write(f"{content}\n\n")
    return output_filepath

# Streamlit App
st.title("Jim Stone Speech Search")

query = st.text_input("Enter a search query:")
if st.button("Search"):
    speeches = load_speeches()
    if query:
        results = search_speeches(query, speeches)
        output_file = save_results(query, results)
        st.success(f"Results saved to: {output_file}")
        for sim, filename, content in results:
            st.write(f"**{filename}** (Similarity: {sim:.2f})")
            st.write(f"Preview: {content[:300]}...")
    else:
        st.error("Please enter a search query.")
