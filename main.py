from flask import Flask, request, jsonify
import PyPDF2
from gensim import corpora, similarities, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import requests
from transformers import pipeline
from googlesearch import search
from textblob import TextBlob
from bs4 import BeautifulSoup
import json
import openai
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

summarizer = pipeline("summarization", model="Falconsai/text_summarization")

def split_text_into_chunks(text, max_chunk_length=512):
    """
    Splits the input text into smaller chunks.

    Args:
        text (str): The input text to be split.
        max_chunk_length (int): Maximum length of each chunk.

    Returns:
        list: List of text chunks.
    """
    chunks = []
    words = text.split()
    current_chunk = ""
    
    for word in words:
        if len(current_chunk) + len(word) < max_chunk_length:
            current_chunk += word + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def summarize_pdf(pdf_path):
    """
    Summarizes the content of a PDF document.

    Args:
        pdf_path: Path to the PDF document.

    Returns:
        str: Summary of the PDF content.
    """
    # Open the PDF
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        # Extract text from each page
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into smaller chunks
        text_chunks = split_text_into_chunks(text)

        # Generate summary for each chunk and concatenate them
        summaries = []
        for chunk in text_chunks:
            input_length = len(chunk.split())
            max_length = min(2 * input_length, 512)  # Adjust the multiplier as needed
            summary = summarizer(chunk, max_length=max_length, min_length=30, do_sample=False)
            summaries.append(summary[0]["summary_text"])
        
    return "\n".join(summaries)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "hello"}), 200

@app.route('/summarize_pdf', methods=['POST'])
def summarize_pdf_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    input_pdf_path = "/tmp/input_pdf.pdf"  # Temporary path to store the uploaded PDF
    file.save(input_pdf_path)

    summary = summarize_pdf(input_pdf_path)
    if summary:
        return jsonify({"summary": summary}), 200
    else:
        return jsonify({"error": "Failed to summarize PDF"}),500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)